# SPDX-License-Identifier: Apache-2.0
"""Runtime MTP head attachment for the mlx-vlm Qwen3.5 (dense) VLM path.

Mirror of ``qwen35_moe_vlm_runtime.py`` for the dense Qwen3.5/3.6 family
(model_type=qwen3_5, e.g. Qwen3.6-27B). The MoE variant was wired up in
PR 1180; this companion handles dense VLM checkpoints that ship MTP
heads (mtp_num_hidden_layers > 0).

It adds:

* a Multi-Token Prediction head (``MTPModule``) to
  ``mlx_vlm.models.qwen3_5.language.LanguageModel`` when the config
  declares ``mtp_num_hidden_layers > 0`` and the checkpoint has MTP
  weights to bind;
* a ``return_hidden=True`` mode on ``LanguageModel.__call__`` that
  returns ``(logits, pre_norm_hidden, gdn_states)``.

Outer ``Model.sanitize`` is already patched separately by
``qwen35_vlm_model.py`` (MTP-key preservation + norm +1 shift), so no
sanitize work is needed here.

The decoder-graph classes (``Qwen3_5DecoderLayer``, ``Qwen3_5Attention``,
``Qwen3_5MLP``, ``Qwen3_5GatedDeltaNet``) are not modified. SSM rollback
on draft rejection uses mlx-vlm's stock
``LanguageModel.rollback_speculative_cache(...)`` which already exists
and consumes the ``gdn_states`` returned from this patched ``__call__``.

Apply ordering: this patch must run *before* ``mlx_vlm.utils.load(...)``
so the patched ``LanguageModel.__init__`` runs. ``maybe_apply_pre_load_patches`` in
``omlx/utils/model_loading.py`` calls ``apply_mlx_vlm_mtp_runtime_patch``
before loading the model, satisfying the ordering for inference. The oQ path in
``omlx/oq.py:_measure_sensitivity`` also calls it before
``vlm_load_model`` for sensitivity measurement.
"""

from __future__ import annotations

import logging
import os
import weakref
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from ..mlx_lm_mtp import prompt_priming, qwen35_dspark

logger = logging.getLogger(__name__)

_APPLIED = False


def _dspark_prefill_capture_enabled() -> bool:
    """Whether prefill forwards should capture DSpark taps for priming.

    On by default. Off leaves the drafter unprimed — still correct, and still
    correctly phased (the seam seeds the context cache at the target's
    absolute offset), just context-starved until generated tokens accumulate.
    """
    return os.environ.get(
        "OMLX_DSPARK_VLM_PREFILL_CAPTURE", "1"
    ).strip().lower() not in ("0", "false", "off")


def apply() -> bool:
    """Apply the mlx-vlm Qwen3.5 (dense) runtime MTP patches. Idempotent."""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        from mlx_vlm.models.qwen3_5 import config as q35_config
        from mlx_vlm.models.qwen3_5 import language as q35_lang
    except Exception as e:
        logger.debug(f"mlx_vlm.qwen3_5 not importable for MTP runtime: {e}")
        return False

    _patch_text_config(q35_config)
    _register_mtp_classes_for_vlm(q35_lang)
    qwen35_dspark.register(q35_lang)
    _patch_vlm_language_model(q35_lang)
    _patch_inner_model_capture(q35_lang)
    # VLMModelAdapter pass-throughs are installed by the MoE runtime patch
    # too; the function is idempotent so calling it twice is safe.
    _patch_vlm_model_adapter()
    _patch_vlm_outer_model_load_weights()

    _APPLIED = True
    logger.info("mlx-vlm Qwen3.5 (dense) runtime MTP patch applied")
    return True


def _patch_vlm_outer_model_load_weights() -> None:
    """Repair miscalibrated MTP-head norms in ``Model.load_weights`` payloads.

    Older oQ conversions stored the head's q_norm/k_norm/mtp.norm one below
    the correct MLX value; ``repair_legacy_head_norms`` re-applies the +1 at
    load time. Idempotent-safe on correctly-converted models.
    """
    try:
        from mlx_vlm.models import qwen3_5 as q35_outer
    except Exception as e:
        logger.debug(f"mlx_vlm outer qwen3_5 not importable: {e}")
        return

    cls = q35_outer.Model
    if getattr(cls, "_omlx_mtp_norm_repair_patched", False):
        return

    original_load_weights = cls.load_weights

    def load_weights(self, weights, strict=True):
        from ..mlx_lm_mtp.norm_repair import repair_legacy_head_norms

        try:
            weights, _ = repair_legacy_head_norms(weights)
        except Exception:
            logger.warning("MTP head-norm repair failed", exc_info=True)
        return original_load_weights(self, weights, strict=strict)

    cls.load_weights = load_weights
    cls._omlx_mtp_norm_repair_patched = True


# ---------------------------------------------------------------------------
# TextConfig — retain mtp_num_hidden_layers as instance attribute.
# ---------------------------------------------------------------------------

def _patch_text_config(q35_config: Any) -> None:
    """Wrap ``TextConfig.from_dict`` so ``mtp_num_hidden_layers`` survives.

    mlx-vlm's ``BaseModelConfig.from_dict`` filters incoming params by the
    dataclass signature, dropping any key that isn't a declared field —
    including ``mtp_num_hidden_layers``. Without it the MTP head can't be
    sized; with it, ``LanguageModel.__init__`` knows to attach a head.
    """
    cls = q35_config.TextConfig
    if getattr(cls, "_omlx_mtp_from_dict_patched", False):
        return

    original_from_dict = cls.from_dict.__func__  # unwrap classmethod

    def patched_from_dict(cls_inner, params):
        instance = original_from_dict(cls_inner, params)
        if params:
            instance.mtp_num_hidden_layers = int(
                params.get("mtp_num_hidden_layers", 0) or 0
            )
        else:
            instance.mtp_num_hidden_layers = 0
        # Embedded-DSpark declaration + the drafter's own architecture, which
        # differs from the trunk in every dimension that matters.
        qwen35_dspark.carry_config(instance, params)
        return instance

    cls.from_dict = classmethod(patched_from_dict)
    cls._omlx_mtp_from_dict_patched = True


# ---------------------------------------------------------------------------
# MTPDecoderLayer + MTPModule — dense VLM classes.
# ---------------------------------------------------------------------------

def _register_mtp_classes_for_vlm(q35_lang: Any) -> None:
    """Attach ``MTPDecoderLayer`` / ``MTPModule`` to the mlx-vlm qwen3_5
    language module. Dense uses ``Qwen3_5MLP`` (no MoE branch)."""
    if hasattr(q35_lang, "MTPModule"):
        return

    Attention = q35_lang.Qwen3_5Attention
    MLP = q35_lang.Qwen3_5MLP
    from mlx_vlm.models.qwen3_5.language import create_attention_mask

    class MTPDecoderLayer(nn.Module):
        """Full-attention transformer layer used inside the dense MTP head."""

        def __init__(self, args):
            super().__init__()
            self.self_attn = Attention(args)
            self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
            self.post_attention_layernorm = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.mlp = MLP(args.hidden_size, args.intermediate_size)

        def __call__(self, x, mask=None, cache=None, position_ids=None):
            r = self.self_attn(self.input_layernorm(x), mask, cache, position_ids)
            h = x + r
            return h + self.mlp(self.post_attention_layernorm(h))

    class MTPModule(nn.Module):
        """Multi-Token Prediction head (mlx-lm PR 990) for dense VLM Qwen3.5/3.6.

        Predicts token t+2 by fusing the backbone pre-norm hidden state at
        position t with the embedding of the sampled main token t+1.
        """

        def __init__(self, args):
            super().__init__()
            self.pre_fc_norm_hidden = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.pre_fc_norm_embedding = nn.RMSNorm(
                args.hidden_size, eps=args.rms_norm_eps
            )
            self.fc = nn.Linear(args.hidden_size * 2, args.hidden_size, bias=False)
            self.layers = [
                MTPDecoderLayer(args) for _ in range(args.mtp_num_hidden_layers)
            ]
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

        def __call__(self, hidden_states, next_token_ids, embed_tokens, cache=None):
            embeds = embed_tokens(next_token_ids)
            e = self.pre_fc_norm_embedding(embeds)
            h = self.pre_fc_norm_hidden(hidden_states)
            fused = self.fc(mx.concatenate([e, h], axis=-1))

            if cache is None:
                cache = [None] * len(self.layers)

            mask = create_attention_mask(fused, cache[0] if cache else None)
            for layer, c in zip(self.layers, cache):
                fused = layer(fused, mask, c)

            return self.norm(fused)

    q35_lang.MTPDecoderLayer = MTPDecoderLayer
    q35_lang.MTPModule = MTPModule


# ---------------------------------------------------------------------------
# LanguageModel — wrap __init__, support return_hidden, add mtp_forward/cache.
# ---------------------------------------------------------------------------

def _patch_vlm_language_model(q35_lang: Any) -> None:
    cls = q35_lang.LanguageModel
    if "_omlx_mtp_runtime_patched" in cls.__dict__:
        return

    from mlx_lm.models.cache import KVCache

    original_init = cls.__init__
    original_call = cls.__call__

    def __init__(self, args, config=None):
        from . import is_mtp_attach_enabled
        from ..mlx_lm_mtp import is_mtp_active

        original_init(self, args, config)
        # Attach MTPModule when the config declares MTP heads so mlx-vlm's
        # load_weights (which skips Model.sanitize for is_mlx_format
        # checkpoints) can place the persisted mtp.* tensors. Whether MTP
        # speculative decode is actually invoked at inference time is gated
        # downstream by ``mlx_lm_mtp.batch_generator._is_mtp_eligible``,
        # which checks the per-instance ``_omlx_mtp_decode_enabled`` marker.
        #
        # Gated by ``is_mtp_attach_enabled()`` so checkpoints that declare
        # mtp_num_hidden_layers > 0 but ship no mtp.* weights (unsloth
        # Qwen3.6 UD MLX builds, issue #1426) don't fail strict load_weights
        # with "Missing N parameters" and silently downgrade to LLM.
        n_mtp = int(getattr(args, "mtp_num_hidden_layers", 0) or 0)
        is_dspark = qwen35_dspark.is_dspark_config(args)
        attach_enabled = bool(is_mtp_attach_enabled())
        self._omlx_mtp_decode_enabled = bool(
            (n_mtp > 0 or is_dspark) and attach_enabled and is_mtp_active()
        )
        self._omlx_dspark_decode_enabled = bool(
            self._omlx_mtp_decode_enabled and is_dspark
        )
        if (n_mtp > 0 or is_dspark) and attach_enabled:
            self.mtp = (
                qwen35_dspark.DSparkHead(args)
                if is_dspark
                else q35_lang.MTPModule(args)
            )
        if self._omlx_mtp_decode_enabled:
            # Depth-k chained drafting works on this path: mtp_forward
            # supports return_hidden below, and rollback uses mlx-vlm's
            # stock rollback_speculative_cache (native partial accepts).
            from ..mlx_lm_mtp import get_mtp_depth

            self._omlx_mtp_chain = True
            if is_dspark:
                # DSpark drafts the whole block from one head forward, so the
                # per-step chain re-entry is replaced by the dspark dispatch
                # in batch_generator. Its context cache is a committed-only
                # singleton: nothing speculative to trim, and the row-wise
                # batch MTP path cannot model it.
                self._omlx_mtp_head_clone = False
                self._omlx_mtp_rowwise_unsupported = True
                # Still a Qwen3.5 trunk: keep the affine verify-qmm kernel
                # armed (the DSpark opt-out in _call_backbone exists for the
                # DeepSeek-V4 target's own quantized linear path).
                self._omlx_dspark_verify_qmm = True
                self._omlx_mtp_depth = qwen35_dspark.resolve_depth(
                    args, get_mtp_depth()
                )
                logger.info(
                    "Qwen3.5 VLM speculative backend selected: embedded DSpark "
                    "(%d layers, %d taps, block %d, draft width %d)",
                    self.mtp.dspark_args.num_hidden_layers,
                    len(self.mtp.dspark_args.target_layer_ids),
                    self.mtp.dspark_args.block_size,
                    self._omlx_mtp_depth,
                )
            else:
                self._omlx_mtp_depth = get_mtp_depth()
            # Prompt-priming capture runs inside the inner Qwen3_5Model
            # forward, which has no reference back to this LanguageModel
            # (the mtp module / make_mtp_cache live here). A weakref avoids
            # a tracked module cycle in the nn.Module tree.
            self.model._omlx_mtp_prime_host = weakref.ref(self)

    def __call__(self, inputs, inputs_embeds=None, mask=None, cache=None, **kwargs):
        """Backbone forward with optional MTP-cycle return shape.

        With ``return_hidden=True``, returns ``LanguageModelOutput`` with
        pre-norm hidden states for the speculative decode cycle. ``n_confirmed``
        is accepted and discarded — the mlx-vlm path uses post-hoc
        ``rollback_speculative_cache`` instead of a confirmed/draft split.
        """
        return_hidden = kwargs.pop("return_hidden", False)
        return_shared_kv = kwargs.pop("return_shared_kv", False)
        kwargs.pop("n_confirmed", None)
        if not return_hidden:
            return original_call(self, inputs, inputs_embeds, mask, cache, **kwargs)

        # Passing any non-None ``capture_layer_ids`` makes stock
        # ``LanguageModel.__call__`` allocate ``hidden_sink`` AND ``gdn_sink``,
        # both of which the MTP cycle needs. Pop any existing value from kwargs
        # to avoid "got multiple values for keyword argument" when the caller
        # already passed capture_layer_ids.
        kwargs.pop("capture_layer_ids", None)
        # DSpark's head input is the concatenation of trunk taps rather than
        # the final pre-norm hidden; the inner model already supports
        # capturing arbitrary layers, so this is the same forward with a
        # different capture set.
        dspark = bool(getattr(self, "_omlx_dspark_decode_enabled", False))
        if dspark:
            taps = qwen35_dspark.target_layer_ids(self.args)
            capture_ids = sorted(set(taps))
        else:
            capture_ids = [len(self.model.layers) - 1]
        out = original_call(
            self,
            inputs,
            inputs_embeds,
            mask,
            cache,
            capture_layer_ids=capture_ids,
            **kwargs,
        )
        from mlx_vlm.models.base import LanguageModelOutput

        if dspark:
            captured = list(out.hidden_states or ())
            if len(captured) != len(capture_ids):
                raise RuntimeError(
                    "DSpark target tap mismatch: "
                    f"captured={len(captured)}, expected={len(capture_ids)}"
                )
            # ``hidden_sink`` is appended in ascending layer order; ``fc``
            # expects the config's own tap order.
            by_layer = dict(zip(capture_ids, captured))
            head_hidden = mx.concatenate([by_layer[i] for i in taps], axis=-1)
        else:
            head_hidden = out.hidden_states[0]
        return LanguageModelOutput(
            logits=out.logits,
            hidden_states=[head_hidden],
            gdn_states=out.gdn_states,
            shared_kv_states={} if return_shared_kv else None,
        )

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        """MTP-head forward (see mlx_lm_mtp.qwen35_model for the depth-k
        chain contract: return_hidden yields the head's post-norm hidden for
        chaining; logits_keep limits the lm_head to the last N positions)."""
        if getattr(self, "_omlx_dspark_decode_enabled", False):
            logits, hidden = self.dspark_forward(
                hidden_states,
                next_token_ids,
                mtp_cache,
                draft_length=int(next_token_ids.shape[-1]),
            )
            if logits_keep and logits.shape[1] > logits_keep:
                logits = logits[:, -logits_keep:]
                hidden = hidden[:, -logits_keep:]
            if return_hidden:
                return logits, hidden
            return logits

        mtp_out = self.mtp(
            hidden_states,
            next_token_ids,
            self.model.embed_tokens,
            mtp_cache,
        )
        logits_source = mtp_out
        if logits_keep and logits_source.shape[1] > logits_keep:
            logits_source = logits_source[:, -logits_keep:, :]
        if self.args.tie_word_embeddings:
            logits = self.model.embed_tokens.as_linear(logits_source)
        else:
            logits = self.lm_head(logits_source)
        if return_hidden:
            return logits, mtp_out
        return logits

    def make_mtp_cache(self):
        if getattr(self, "_omlx_dspark_decode_enabled", False):
            return self.make_dspark_cache()
        if hasattr(self, "mtp"):
            return [KVCache() for _ in self.mtp.layers]
        return []

    def mtp_take_primed(self, cache, main_token):
        """Own the prompt-to-decode seam when this instance is DSpark-shaped.

        Lightning MTP instances of the same class decline (``NotImplemented``)
        and fall through to ``prompt_priming``'s generic fold.
        """
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            return NotImplemented
        return qwen35_dspark.take_primed(self, cache, main_token)

    cls.__init__ = __init__
    cls.__call__ = __call__
    cls.mtp_forward = mtp_forward
    cls.make_mtp_cache = make_mtp_cache
    cls.mtp_take_primed = mtp_take_primed
    qwen35_dspark.install_host_methods(cls)
    cls._omlx_mtp_runtime_patched = True


# ---------------------------------------------------------------------------
# Inner Qwen3_5Model — prompt-priming capture on prefill/decode forwards.
# ---------------------------------------------------------------------------

def _patch_inner_model_capture(q35_lang: Any) -> None:
    """Wrap ``Qwen3_5Model.__call__`` to fold prompt chunks into the MTP head.

    The inner model's return value is the trunk-normed hidden for every
    position of the forward — exactly what the head history fold needs — and
    scheduler prefill reaches it via the outer ``LanguageModel.__call__``
    delegate, so this single wrap covers external prefill, chunked prefill
    and the seam decode steps.

    Skips: image/recursive calls (``inputs_embeds``), MTP verify and other
    capture forwards (any of ``capture_layer_ids`` / ``hidden_sink`` /
    ``gdn_sink``), unknown extra positional call shapes, and hosts without
    an MTP head (weakref unset). All real gating lives in
    ``prompt_priming.maybe_capture`` and fails safe to unprimed.
    """
    cls = q35_lang.Qwen3_5Model
    if getattr(cls, "_omlx_mtp_prime_capture_patched", False):
        return

    original_call = cls.__call__

    def __call__(
        self, inputs, inputs_embeds=None, mask=None, cache=None, *args, **kwargs
    ):
        eligible = (
            inputs_embeds is None
            and cache is not None
            and not args
            and kwargs.get("capture_layer_ids") is None
            and kwargs.get("hidden_sink") is None
            and kwargs.get("gdn_sink") is None
        )
        host_ref = getattr(self, "_omlx_mtp_prime_host", None) if eligible else None
        host = host_ref() if host_ref is not None else None

        # Embedded DSpark primes from trunk *taps*, which the plain forward
        # does not compute a handle for — ask the inner model to capture them
        # on the way through. Passing a hidden_sink bypasses the single-row
        # batch-cache shortcut above the layer loop, which is why this is
        # kill-switchable: the MTP verify path already runs that same
        # capture-enabled shape on every cycle, but prefill does not.
        dspark_taps = None
        if (
            host is not None
            and getattr(host, "_omlx_dspark_decode_enabled", False)
            and _dspark_prefill_capture_enabled()
        ):
            dspark_taps = qwen35_dspark.target_layer_ids(host.args)

        if dspark_taps:
            capture_ids = sorted(set(dspark_taps))
            sink: list = []
            # The caller passes these three *explicitly as None* on a plain
            # forward (``LanguageModel.__call__`` always forwards them), so
            # the eligibility check above sees None while the keys are still
            # in ``kwargs`` — re-supplying them here without dropping the
            # originals raises "got multiple values for keyword argument".
            # ``gdn_sink`` stays dropped: leaving it None keeps the trunk off
            # its target_verify path for what is an ordinary prefill.
            call_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ("capture_layer_ids", "hidden_sink", "gdn_sink")
            }
            out = original_call(
                self,
                inputs,
                inputs_embeds,
                mask,
                cache,
                *args,
                capture_layer_ids=capture_ids,
                hidden_sink=sink,
                **call_kwargs,
            )
            try:
                if len(sink) == len(capture_ids):
                    by_layer = dict(zip(capture_ids, sink))
                    taps = mx.concatenate(
                        [by_layer[i] for i in dspark_taps], axis=-1
                    )
                    qwen35_dspark.capture_prompt(host, inputs, taps, cache)
                else:
                    qwen35_dspark.drop_primed(host)
            except Exception:
                qwen35_dspark.drop_primed(host)
                logger.debug("DSpark prompt capture failed", exc_info=True)
            return out

        out = original_call(
            self, inputs, inputs_embeds, mask, cache, *args, **kwargs
        )
        # The generic fold is Lightning-MTP-shaped (hidden + next token
        # through mtp_forward); a DSpark host must never enter it.
        if host is not None and not getattr(
            host, "_omlx_dspark_decode_enabled", False
        ):
            try:
                prompt_priming.maybe_capture(host, inputs, out, cache)
            except Exception:
                logger.debug(
                    "MTP prompt-priming capture failed", exc_info=True
                )
        return out

    cls.__call__ = __call__
    cls._omlx_mtp_prime_capture_patched = True


# ---------------------------------------------------------------------------
# VLMModelAdapter — add MTP pass-through methods at runtime.
# ---------------------------------------------------------------------------

def _patch_vlm_model_adapter() -> None:
    """Extend ``omlx.models.vlm.VLMModelAdapter`` with MTP plumbing.

    Same setup as the MoE runtime patch — idempotent, so calling from
    both dense and MoE apply() is safe.
    """
    try:
        from omlx.models.vlm import VLMModelAdapter
    except Exception as e:
        logger.debug(f"VLMModelAdapter not importable: {e}")
        return

    if getattr(VLMModelAdapter, "_omlx_mtp_adapter_patched", False):
        return

    @property
    def mtp(self):
        return getattr(self._language_model, "mtp", None)

    def mtp_forward(
        self,
        hidden_states,
        next_token_ids,
        mtp_cache,
        return_hidden: bool = False,
        logits_keep: int = 0,
    ):
        # Forward the depth-k chain kwargs only when set so language models
        # whose mtp_forward predates them (MoE runtime) keep working.
        if return_hidden or logits_keep:
            return self._language_model.mtp_forward(
                hidden_states,
                next_token_ids,
                mtp_cache,
                return_hidden=return_hidden,
                logits_keep=logits_keep,
            )
        return self._language_model.mtp_forward(
            hidden_states, next_token_ids, mtp_cache
        )

    def make_mtp_cache(self):
        if hasattr(self._language_model, "make_mtp_cache"):
            return self._language_model.make_mtp_cache()
        return []

    def rollback_speculative_cache(self, caches, gdn_states, accepted, block_size):
        return self._language_model.rollback_speculative_cache(
            caches, gdn_states, accepted, block_size
        )

    VLMModelAdapter.mtp = mtp
    VLMModelAdapter.mtp_forward = mtp_forward
    VLMModelAdapter.make_mtp_cache = make_mtp_cache
    VLMModelAdapter.rollback_speculative_cache = rollback_speculative_cache
    VLMModelAdapter._omlx_mtp_adapter_patched = True
