# SPDX-License-Identifier: Apache-2.0
"""Embedded DSpark drafter for Qwen3.5/3.6 targets (SpecForge/DFlash topology).

Companion to :mod:`deepseek_v4_dspark`, but for a *different* drafter shape.
DeepSeek's 0731 checkpoint stores a DeepSpec-style DSpark head as N sequential
*stages* under ``mtp.0..N-1``, each consuming one target-layer tap. RadixArk's
``Qwen3.8-27B-DSpark`` is SpecForge-shaped instead (``specforge`` ->
``DFlashDraftModel`` + ``DSparkDraftModel``):

* **one** ``fc`` projection over all taps *concatenated*
  (``Linear(len(target_layer_ids) * hidden, hidden)``), followed by a single
  ``hidden_norm``; the result is the shared *context* stream, and
* **N plain Qwen3 decoder layers** (GQA, q_norm/k_norm, SwiGLU) that each
  attend, non-causally, over ``[context ; draft block]`` — dual-source KV:
  every layer runs its own ``k_proj``/``v_proj`` over the *same* projected
  context, so the context K/V is per-layer and cached per-layer.

``deepseek_v4_dspark.DSparkBlock`` therefore cannot be reused. What *is*
shared is the runtime contract: the native-MTP ``GenerationBatch`` integration
in :mod:`batch_generator` owns verification, exact rejection sampling,
rollback and delivery, and reaches the head through five duck-typed members on
the host language model::

    _omlx_dspark_decode_enabled = True     # host discovery (_dspark_host)
    args.dspark_block_size                 # caps draft depth
    make_mtp_cache()                       # list of per-layer context caches
    dspark_append_context(taps, cache, *, start_offset=None)
    dspark_forward(taps, anchor_ids, cache, *, draft_length=n)
    dspark_markov(token_ids)

Block convention (load-bearing, and *not* the same as DeepSeek's). Upstream
``dflash.py::spec_generate`` builds a block of ``block_size`` positions whose
first entry is the newest target-confirmed token and whose remaining entries
are the MASK/noise token, then reads logits from ``[:, -block_size + 1:, :]``
and writes them back to positions ``1..block_size-1``. It is a masked-diffusion
head: the logits at a MASK position predict the token *at that position*, not
the next one. So a block of width ``W`` yields ``W - 1`` drafts, and
``dspark_forward(draft_length=k)`` runs ``k + 1`` query positions and returns
``logits[:, 1:]``. Hence the usable draft depth is ``dspark_block_size - 1``.

Positions are absolute: a context row for the token at position ``p`` is
RoPE'd at ``p``, and the draft block occupies ``offset .. offset + W - 1``
where ``offset`` is the context cache's end — which is exactly the position of
the anchor token, since the anchor's own context row is the newest one
committed. Keeping the cache offset absolute is what lets the drafter work at
the right rotary phase even when the prompt was never primed.

The context cache is committed-only: draft-block K/V is recomputed per cycle
and never written back, so a rejected draft needs no rollback here — the
target cache's rollback is the only one that matters.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

_PRIME_CTX_ATTR = "_omlx_dspark_prime_ctx"

#: ``config.json`` keys written by ``tools/graft_dspark.py``. Both mlx-lm's
#: ``TextModelArgs.from_dict`` and mlx-vlm's ``TextConfig.from_dict`` filter
#: incoming params down to declared dataclass fields, so these have to be
#: re-attached to the instance by hand (same trick the patches already use for
#: ``mtp_num_hidden_layers``).
CONFIG_KEYS = (
    "dspark_block_size",
    "dspark_target_layer_ids",
    "dspark_markov_rank",
    "dspark_noise_token_id",
    "dspark_num_hidden_layers",
    "dspark_num_attention_heads",
    "dspark_num_key_value_heads",
    "dspark_head_dim",
    "dspark_intermediate_size",
    "dspark_rms_norm_eps",
    "dspark_rope_parameters",
    "dspark_confidence_head",
    "dspark_confidence_with_markov",
)


def carry_config(instance: Any, params: Optional[dict]) -> None:
    """Re-attach ``dspark_*`` config keys that ``from_dict`` filtered out."""
    params = params or {}
    for key in CONFIG_KEYS:
        if key in params:
            setattr(instance, key, params[key])


def is_dspark_config(config: Any) -> bool:
    """True when *config* declares an embedded DSpark drafter."""
    block_size = int(getattr(config, "dspark_block_size", 0) or 0)
    target_ids = tuple(getattr(config, "dspark_target_layer_ids", ()) or ())
    return block_size > 0 and bool(target_ids)


def target_layer_ids(config: Any) -> tuple[int, ...]:
    return tuple(int(i) for i in (getattr(config, "dspark_target_layer_ids", ()) or ()))


def max_draft_length(config: Any) -> int:
    """Largest draft depth the trained block width supports.

    One block position is spent on the anchor token, so a ``block_size`` of 7
    drafts 6 tokens. ``dspark_forward`` will honour a longer request (nothing
    in the graph forbids it — it is only outside the training distribution),
    but the host clamps ``_omlx_mtp_depth`` to this by default.
    """
    return max(1, int(getattr(config, "dspark_block_size", 1) or 1) - 1)


def resolve_depth(config: Any, requested: int) -> int:
    """Draft depth for a DSpark head, given the global MTP depth setting.

    Block drafting costs one head forward regardless of width, so a depth of
    1 — the process-wide default when nothing set it — would pay DSpark's
    whole price for a single token and land below the Lightning MTP head it
    replaces. Treat an unset depth as "use the trained block", and let the
    adaptive depth controller narrow it from there; an explicit depth > 1 is
    still honoured as a cap.
    """
    ceiling = max_draft_length(config)
    requested = int(requested or 0)
    if requested <= 1:
        return ceiling
    return min(requested, ceiling)


# ---------------------------------------------------------------------------
# Normalised view over the host's args object.
# ---------------------------------------------------------------------------


@dataclass
class DSparkArgs:
    hidden_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_parameters: Optional[dict]
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    markov_rank: int
    noise_token_id: int
    confidence_head: bool
    confidence_with_markov: bool


def dspark_args(config: Any) -> DSparkArgs:
    """Build the drafter's own architecture args from the target's config.

    Everything the drafter needs is namespaced ``dspark_*`` so it can differ
    from the target trunk — and on Qwen3.8 it does, in every dimension that
    matters: the trunk is a hybrid linear/full-attention stack with
    ``head_dim=256`` and interleaved MRoPE, the drafter is five plain Qwen3
    dense layers with ``head_dim=128`` and YaRN.
    """
    taps = target_layer_ids(config)
    if not taps:
        raise ValueError("DSpark requires a non-empty dspark_target_layer_ids")
    hidden = int(config.hidden_size)
    heads = int(getattr(config, "dspark_num_attention_heads", 0) or 0)
    head_dim = int(getattr(config, "dspark_head_dim", 0) or 0)
    if heads <= 0:
        raise ValueError("DSpark requires dspark_num_attention_heads")
    if head_dim <= 0:
        head_dim = hidden // heads
    kv_heads = int(getattr(config, "dspark_num_key_value_heads", 0) or heads)
    if kv_heads <= 0 or heads % kv_heads != 0:
        raise ValueError(
            "DSpark requires num_attention_heads divisible by num_key_value_heads, "
            f"got {heads} / {kv_heads}"
        )
    return DSparkArgs(
        hidden_size=hidden,
        vocab_size=int(config.vocab_size),
        num_hidden_layers=int(getattr(config, "dspark_num_hidden_layers", 0) or 0)
        or len(taps),
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        intermediate_size=int(getattr(config, "dspark_intermediate_size", 0) or 0)
        or 4 * hidden,
        rms_norm_eps=float(
            getattr(config, "dspark_rms_norm_eps", None)
            or getattr(config, "rms_norm_eps", 1e-6)
        ),
        rope_parameters=getattr(config, "dspark_rope_parameters", None),
        max_position_embeddings=int(
            getattr(config, "max_position_embeddings", 0) or 32768
        ),
        block_size=int(getattr(config, "dspark_block_size", 1) or 1),
        target_layer_ids=taps,
        markov_rank=int(getattr(config, "dspark_markov_rank", 0) or 0),
        noise_token_id=int(getattr(config, "dspark_noise_token_id", 0) or 0),
        confidence_head=bool(getattr(config, "dspark_confidence_head", True)),
        confidence_with_markov=bool(
            getattr(config, "dspark_confidence_with_markov", True)
        ),
    )


# ---------------------------------------------------------------------------
# Context cache — committed-only, absolute-offset, per drafter layer.
# ---------------------------------------------------------------------------


class DSparkContextCache:
    """Growing K/V buffer for one drafter layer's view of the target context.

    Only target-confirmed rows are ever appended, so the cache always sits on
    the committed timeline and draft rejection never touches it. ``offset`` is
    the *absolute* token position after the newest stored row; the number of
    physically stored rows can be smaller when the cache was seeded mid-stream
    (an unprimed activation keeps the rotary phase correct without inventing
    the prompt's hidden states).
    """

    def __init__(self, max_size: int = 0, step: int = 256):
        self.step = int(step)
        #: 0 keeps every committed row; otherwise the newest ``max_size``.
        self.max_size = max(0, int(max_size))
        self.keys: Optional[mx.array] = None
        self.values: Optional[mx.array] = None
        self.offset = 0
        self._len = 0

    # -- lifecycle ---------------------------------------------------------
    def seed_offset(self, offset: int) -> None:
        """Start an empty cache at an absolute position."""
        if self._len:
            raise ValueError("cannot seed a non-empty DSpark context cache")
        self.offset = int(offset)

    @property
    def state(self) -> tuple[Optional[mx.array], Optional[mx.array]]:
        if not self._len:
            return None, None
        return self.keys[:, :, : self._len], self.values[:, :, : self._len]

    def __len__(self) -> int:
        return self._len

    def _evict(self, incoming: int) -> None:
        """Drop the oldest rows so ``max_size`` still holds after the append.

        Compaction copies into a fresh buffer with ``step`` rows of headroom,
        so a full cache pays it once every ``step`` decode steps rather than
        on every token. ``offset`` is untouched: it tracks absolute position,
        not stored rows, and the surviving rows keep the rotary phase they
        were written with.
        """
        if not self.max_size or self.keys is None:
            return
        if self._len + incoming <= self.max_size + self.step:
            return
        keep = max(0, self.max_size - incoming)
        capacity = self.max_size + self.step
        shape = (self.keys.shape[0], self.keys.shape[1], capacity, self.keys.shape[3])
        new_keys = mx.zeros(shape, self.keys.dtype)
        new_values = mx.zeros(shape, self.values.dtype)
        if keep:
            new_keys[:, :, :keep] = self.keys[:, :, self._len - keep : self._len]
            new_values[:, :, :keep] = self.values[:, :, self._len - keep : self._len]
        self.keys = new_keys
        self.values = new_values
        self._len = keep

    def append(
        self,
        keys: mx.array,
        values: mx.array,
        *,
        start_offset: Optional[int] = None,
    ) -> None:
        n = int(keys.shape[2])
        if start_offset is not None:
            start = int(start_offset)
            if self._len == 0 and self.keys is None:
                self.offset = start
            elif self.offset != start:
                raise ValueError(
                    "DSpark context is not contiguous: "
                    f"cache={self.offset}, append={start}"
                )
        if n == 0:
            return
        advance = n
        if self.max_size and n > self.max_size:
            # A prefill chunk longer than the whole window: keep only its
            # newest rows. ``offset`` still advances by the full chunk — it is
            # an absolute position, and the rows we keep were already rotated
            # at theirs.
            drop = n - self.max_size
            keys = keys[:, :, drop:]
            values = values[:, :, drop:]
            n = self.max_size
        self._evict(n)
        prev = self._len
        if self.keys is None:
            cap = self.step * ((n + self.step - 1) // self.step)
            shape = (keys.shape[0], keys.shape[1], cap, keys.shape[3])
            self.keys = mx.zeros(shape, keys.dtype)
            self.values = mx.zeros(
                (values.shape[0], values.shape[1], cap, values.shape[3]),
                values.dtype,
            )
        elif prev + n > self.keys.shape[2]:
            extra = self.step * ((n + self.step - 1) // self.step)
            pad_k = mx.zeros(
                (self.keys.shape[0], self.keys.shape[1], extra, self.keys.shape[3]),
                self.keys.dtype,
            )
            pad_v = mx.zeros(
                (
                    self.values.shape[0],
                    self.values.shape[1],
                    extra,
                    self.values.shape[3],
                ),
                self.values.dtype,
            )
            self.keys = mx.concatenate([self.keys[:, :, :prev], pad_k], axis=2)
            self.values = mx.concatenate([self.values[:, :, :prev], pad_v], axis=2)
        self.keys[:, :, prev : prev + n] = keys
        self.values[:, :, prev : prev + n] = values
        self._len = prev + n
        self.offset += advance

    # -- rollback ----------------------------------------------------------
    def is_trimmable(self) -> bool:
        return True

    def trim(self, n: int) -> int:
        """Drop the newest ``n`` rows (never used on the committed path)."""
        n = min(int(n), self._len)
        if n <= 0:
            return 0
        self._len -= n
        self.offset -= n
        return n


# ---------------------------------------------------------------------------
# Modules. Names mirror the checkpoint exactly:
#
#   mtp.fc.weight                         [hidden, taps * hidden]
#   mtp.hidden_norm.weight                [hidden]
#   mtp.layers.<i>.self_attn.{q,k,v,o}_proj.weight
#   mtp.layers.<i>.self_attn.{q,k}_norm.weight
#   mtp.layers.<i>.mlp.{gate,up,down}_proj.weight
#   mtp.layers.<i>.{input_layernorm,post_attention_layernorm}.weight
#   mtp.norm.weight
#   mtp.markov_head.markov_w{1,2}.weight
#   mtp.confidence_head.proj.{weight,bias}
# ---------------------------------------------------------------------------


def _build_rope(args: DSparkArgs):
    """Rotary embedding for the drafter's own rope_parameters.

    RadixArk trained against ``Qwen/Qwen3.8-27B-FP8``'s YaRN schedule
    (theta 1e7, factor 32, beta_fast 32 / beta_slow 1, original context 8192).
    mlx-lm's ``YarnRoPE`` folds the ``0.1 * ln(factor) + 1`` attention factor
    into the input scale, which is equivalent to transformers scaling cos/sin.
    """
    params = dict(args.rope_parameters or {})
    base = float(params.get("rope_theta", 10000.0))
    try:
        from mlx_lm.models.rope_utils import initialize_rope

        return initialize_rope(
            args.head_dim,
            base=base,
            traditional=False,
            scaling_config=params or None,
            max_position_embeddings=args.max_position_embeddings,
        )
    except Exception:
        logger.warning(
            "DSpark: falling back to unscaled RoPE (base=%s); "
            "drafter rope_parameters were not usable",
            base,
            exc_info=True,
        )
        return nn.RoPE(args.head_dim, traditional=False, base=base)


class DSparkAttention(nn.Module):
    """Dual-source, non-causal GQA over ``[committed context ; draft block]``.

    Queries come only from the draft block. Keys/values come from both the
    projected target context (cached, one entry per committed token) and the
    block itself. There is no causal mask: the block is predicted in parallel,
    every position seeing every other (``Qwen3DFlashAttention.is_causal`` is
    ``False`` upstream and ``spec_generate`` passes ``is_causal=False``).
    """

    def __init__(self, args: DSparkArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = _build_rope(args)

    def _kv(self, x: mx.array, offset: int) -> tuple[mx.array, mx.array]:
        b, length, _ = x.shape
        keys = self.k_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        keys = self.k_norm(keys).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        values = values.transpose(0, 2, 1, 3)
        return self.rope(keys, offset=offset), values

    def append_context(
        self,
        context: mx.array,
        cache: DSparkContextCache,
        *,
        start_offset: Optional[int] = None,
    ) -> None:
        """Project + cache committed context rows at their absolute positions."""
        offset = cache.offset if start_offset is None else int(start_offset)
        keys, values = self._kv(context, offset)
        cache.append(keys, values, start_offset=start_offset)

    def __call__(self, x: mx.array, cache: DSparkContextCache) -> mx.array:
        b, width, _ = x.shape
        offset = cache.offset

        queries = self.q_proj(x).reshape(b, width, self.n_heads, self.head_dim)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)

        keys, values = self._kv(x, offset)
        ctx_keys, ctx_values = cache.state
        if ctx_keys is not None:
            keys = mx.concatenate([ctx_keys, keys], axis=2)
            values = mx.concatenate([ctx_values, values], axis=2)

        out = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=None
        )
        out = out.transpose(0, 2, 1, 3).reshape(b, width, -1)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    def __init__(self, args: DSparkArgs):
        super().__init__()
        dim, hidden = args.hidden_size, args.intermediate_size
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.self_attn = DSparkAttention(args)
        self.mlp = DSparkMLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(self, x: mx.array, cache: DSparkContextCache) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DSparkMarkovHead(nn.Module):
    """Low-rank learned bigram bias on the draft logits (DeepSpec vanilla)."""

    def __init__(self, args: DSparkArgs):
        super().__init__()
        self.markov_w1 = nn.Embedding(args.vocab_size, args.markov_rank)
        self.markov_w2 = nn.Linear(args.markov_rank, args.vocab_size, bias=False)

    def __call__(self, token_ids: mx.array) -> tuple[mx.array, mx.array]:
        embedding = self.markov_w1(token_ids)
        return self.markov_w2(embedding), embedding


class DSparkConfidenceHead(nn.Module):
    """Per-position accept-rate predictor (``AcceptRatePredictor``).

    Unused by the current draft/verify loop — the depth controller in
    :mod:`batch_generator` adapts on measured accept rates instead — but it is
    constructed so strict ``load_weights`` binds the checkpoint's tensors.
    Note the drafter ships ``proj.bias``: ``nn.Linear(dim, 1)`` in torch is
    biased by default, unlike DeepSeek's DSpark confidence head.
    """

    def __init__(self, width: int):
        super().__init__()
        self.proj = nn.Linear(width, 1, bias=True)

    def __call__(self, features: mx.array) -> mx.array:
        return self.proj(features).squeeze(-1)


class DSparkHead(nn.Module):
    """The whole drafter, stored under ``<language_model.>mtp.``."""

    def __init__(self, config: Any):
        super().__init__()
        args = dspark_args(config)
        self.dspark_args = args
        self.fc = nn.Linear(
            len(args.target_layer_ids) * args.hidden_size,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.layers = [DSparkDecoderLayer(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        # Attributes are only bound when the checkpoint ships them: an unset
        # sub-module keeps the parameter tree exactly matching the weights,
        # which is what strict ``load_weights`` compares against.
        if args.markov_rank > 0:
            self.markov_head = DSparkMarkovHead(args)
        if args.confidence_head:
            width = args.hidden_size + (
                args.markov_rank if args.confidence_with_markov else 0
            )
            self.confidence_head = DSparkConfidenceHead(width)

    # -- context -----------------------------------------------------------
    def make_cache(self) -> list[DSparkContextCache]:
        window = context_window()
        return [DSparkContextCache(max_size=window) for _ in self.layers]

    def project_context(self, taps: mx.array) -> mx.array:
        """``hidden_norm(fc(concat(taps)))`` — shared by every layer."""
        return self.hidden_norm(self.fc(taps))

    def append_context(
        self,
        taps: mx.array,
        cache: Sequence[DSparkContextCache],
        *,
        start_offset: Optional[int] = None,
    ) -> mx.array:
        context = self.project_context(taps)
        for layer, layer_cache in zip(self.layers, cache):
            layer.self_attn.append_context(
                context, layer_cache, start_offset=start_offset
            )
        return context

    # -- block -------------------------------------------------------------
    def __call__(
        self, block_embeds: mx.array, cache: Sequence[DSparkContextCache]
    ) -> mx.array:
        h = block_embeds
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(h, layer_cache)
        return self.norm(h)


# ---------------------------------------------------------------------------
# Host methods. Installed onto the patched language-model class (mlx-lm's
# ``TextModel`` or mlx-vlm's ``LanguageModel``); both expose ``args``,
# ``model.embed_tokens``, ``lm_head`` and carry the head as ``self.mtp``.
# ---------------------------------------------------------------------------


def _head_logits(host: Any, hidden: mx.array) -> mx.array:
    """Project through the *target's* head — the drafter ships neither."""
    if getattr(host.args, "tie_word_embeddings", False):
        return host.model.embed_tokens.as_linear(hidden)
    return host.lm_head(hidden)


def install_host_methods(cls: type) -> None:
    """Add the ``dspark_*`` contract to a patched language-model class."""
    if getattr(cls, "_omlx_dspark_host_patched", False):
        return

    def make_dspark_cache(self):
        return self.mtp.make_cache()

    def dspark_append_context(self, main_hidden, cache, *, start_offset=None):
        """Commit target taps for tokens the target cache already holds."""
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            raise RuntimeError("DSpark context requested on a non-DSpark model")
        return self.mtp.append_context(
            main_hidden, cache, start_offset=start_offset
        )

    def dspark_forward(self, main_hidden, anchor_ids, cache=None, *, draft_length=None):
        """Append committed taps, then draft one block in parallel.

        ``main_hidden`` is the concatenated target taps for newly committed
        tokens, ``anchor_ids`` the newest target-confirmed token. Returns
        ``(logits[B, k, V], hidden[B, k, H])`` for the ``k`` MASK positions —
        ``logits[:, i]`` is the distribution of draft ``i``.
        """
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            raise RuntimeError("DSpark forward requested on a non-DSpark model")
        if cache is None:
            cache = self.make_mtp_cache()
        if main_hidden is not None and int(main_hidden.shape[1]) > 0:
            self.dspark_append_context(main_hidden, cache)

        args = self.mtp.dspark_args
        width = int(draft_length or getattr(self, "_omlx_mtp_depth", 1) or 1)
        width = max(1, width)
        anchor = anchor_ids.reshape(anchor_ids.shape[0], -1)[:, -1:]
        noise = mx.full(
            (anchor.shape[0], width), args.noise_token_id, dtype=anchor.dtype
        )
        block_ids = mx.concatenate([anchor, noise], axis=1)

        hidden = self.mtp(self.model.embed_tokens(block_ids), cache)
        # Masked-diffusion head: position 0 carries the known anchor and its
        # output is discarded; every MASK position predicts its own token.
        hidden = hidden[:, 1:]
        return _head_logits(self, hidden), hidden

    def dspark_markov(self, token_ids):
        """Previous-token logit bias plus its rank-R embedding."""
        head = getattr(self.mtp, "markov_head", None)
        if head is None:
            zeros = mx.zeros((int(token_ids.size), 1), dtype=mx.float32)
            return zeros, zeros
        return head(token_ids)

    def dspark_confidence(self, hidden, markov_embedding=None):
        """Predicted per-position accept rate (unused by the current loop)."""
        head = getattr(self.mtp, "confidence_head", None)
        if head is None:
            return None
        features = hidden
        if markov_embedding is not None and self.mtp.dspark_args.confidence_with_markov:
            features = mx.concatenate([hidden, markov_embedding], axis=-1)
        return mx.sigmoid(head(features))

    def dspark_calibration_forward(self, target_hiddens, input_ids):
        """Exercise every DSpark linear once, for oQe imatrix collection."""
        if not getattr(self, "_omlx_dspark_decode_enabled", False):
            return None
        width = min(
            max_draft_length(self.args), max(1, int(input_ids.shape[1]) - 1)
        )
        logits, _ = dspark_forward(
            self,
            target_hiddens[:, :-1],
            input_ids[:, -1:],
            self.make_mtp_cache(),
            draft_length=width,
        )
        mx.eval(logits)
        return logits

    cls.make_dspark_cache = make_dspark_cache
    cls.dspark_append_context = dspark_append_context
    cls.dspark_forward = dspark_forward
    cls.dspark_markov = dspark_markov
    cls.dspark_confidence = dspark_confidence
    cls.dspark_calibration_forward = dspark_calibration_forward
    cls._omlx_dspark_host_patched = True


# ---------------------------------------------------------------------------
# Prompt priming — stream prefill taps into the context cache.
#
# Same fail-safe shape as ``prompt_priming`` / ``deepseek_v4_dspark``: a single
# slot on the host, invalidated by any offset discontinuity (rewind, trim,
# request switch), degrading to an unprimed-but-correctly-phased cache rather
# than to a wrong history.
# ---------------------------------------------------------------------------


@dataclass
class _PrimeCtx:
    caches: list
    expected_target_offset: int
    chunks: int = 0


def _ctx_rows(caches: Sequence[Any]) -> int:
    """Committed context rows the drafter actually holds."""
    return min((len(c) for c in caches), default=0)


#: Committed context rows the drafter keeps, oldest dropped first. 0 = keep
#: everything. The drafter's own config declares no sliding window, but that
#: describes its *architecture*, not the sequence lengths SpecForge trained it
#: on — and measured acceptance on this checkpoint collapses from 46% at a
#: 1k-token context to 0% at 16k on the same corpus, with the context verified
#: complete and correctly phased. A suffix window keeps the drafter inside the
#: range it behaves in; it also bounds the cache (5 layers x 8 kv heads x 128
#: dims x 2 for k/v is ~20 KB per row). Rotary positions stay absolute, so
#: windowing changes what the drafter sees, never the phase it sees it at.
_DEFAULT_CONTEXT_WINDOW = 4096


def context_window() -> int:
    """Max committed context rows per drafter layer (0 = unbounded)."""
    raw = os.environ.get("OMLX_DSPARK_CONTEXT_WINDOW")
    if raw is None:
        return _DEFAULT_CONTEXT_WINDOW
    try:
        return max(0, int(raw))
    except ValueError:
        return _DEFAULT_CONTEXT_WINDOW


def priming_enabled() -> bool:
    return os.environ.get("OMLX_DSPARK_PROMPT_PRIMING", "1").strip().lower() not in (
        "0",
        "false",
        "off",
    )


def _target_cache_offset(cache: Optional[Sequence[Any]]) -> Optional[int]:
    """First readable attention-layer offset, tolerant of batch caches."""
    if not cache:
        return None

    def read(entry: Any) -> Optional[int]:
        offset = getattr(entry, "offset", None)
        if type(offset) is int:
            return offset
        if offset is not None and getattr(offset, "size", 0) == 1:
            try:
                return int(offset.reshape(()).item())
            except Exception:
                return None
        return None

    for entry in cache:
        got = read(entry)
        if got is not None:
            return got
        for child in getattr(entry, "caches", ()) or ():
            got = read(child)
            if got is not None:
                return got
    return None


def drop_primed(host: Any) -> None:
    try:
        delattr(host, _PRIME_CTX_ATTR)
    except AttributeError:
        pass


def capture_prompt(
    host: Any,
    inputs: mx.array,
    taps: mx.array,
    target_cache: Optional[Sequence[Any]],
) -> None:
    """Fold one prefill chunk's target taps into the DSpark context cache."""
    if not priming_enabled():
        return
    if target_cache is None or getattr(inputs, "ndim", 0) != 2 or inputs.shape[0] != 1:
        return
    offset_after = _target_cache_offset(target_cache)
    if offset_after is None:
        return
    seq_len = int(inputs.shape[1])
    start_offset = offset_after - seq_len
    ctx = getattr(host, _PRIME_CTX_ATTR, None)
    if not isinstance(ctx, _PrimeCtx) or ctx.expected_target_offset != start_offset:
        if isinstance(ctx, _PrimeCtx):
            # A live context did not continue: rewind, trim, request switch, or
            # a prefill chunk arriving out of step with the target cache. The
            # rows collected so far are discarded rather than spliced, so a
            # long prompt can end up primed from only its tail — visible here
            # and in the ``rows`` vs ``offset`` gap logged at the seam.
            logger.info(
                "DSpark priming restarted after %d chunk(s): target offset %d, "
                "expected %d — discarding %d context row(s)",
                ctx.chunks,
                start_offset,
                ctx.expected_target_offset,
                _ctx_rows(ctx.caches),
            )
        drop_primed(host)
        if seq_len <= 1:
            # A lone decode step cannot start a prompt timeline, and building
            # one would tax every standard (non-MTP) step with five layers of
            # k/v projection for context that will never be used.
            return
        ctx = _PrimeCtx(
            caches=host.make_dspark_cache(),
            expected_target_offset=start_offset,
        )
        setattr(host, _PRIME_CTX_ATTR, ctx)
    try:
        host.dspark_append_context(taps, ctx.caches, start_offset=start_offset)
    except Exception:
        drop_primed(host)
        logger.debug("DSpark prompt capture failed", exc_info=True)
        return
    ctx.expected_target_offset = offset_after
    ctx.chunks += 1
    arrays = []
    for cache in ctx.caches:
        if cache.keys is not None:
            arrays.append(cache.keys)
            arrays.append(cache.values)
    if arrays:
        mx.async_eval(arrays)


def take_primed(
    host: Any,
    target_cache: Optional[Sequence[Any]],
    _main_token: Any,
) -> Optional[tuple[list, int]]:
    """Close the prompt-to-decode seam for ``_post_init_mtp``.

    Returns ``(caches, history_offset)``. When priming did not survive (or
    never ran), an *empty* cache seeded at the right absolute position is
    returned instead of ``None``: the drafter then starts context-starved but
    at the correct rotary phase, and fills in as tokens commit. Returning
    ``None`` would leave the caller with an offset-0 cache and every
    subsequent context row rotated to the wrong position.
    """
    ctx = getattr(host, _PRIME_CTX_ATTR, None)
    drop_primed(host)
    target_offset = _target_cache_offset(target_cache)
    if target_offset is None:
        return None
    # The activation forward has already pushed one target token through with
    # capture disabled, so a live context should end exactly one short.
    if isinstance(ctx, _PrimeCtx) and ctx.expected_target_offset == target_offset - 1:
        history_offset = min((c.offset for c in ctx.caches), default=0)
        rows = _ctx_rows(ctx.caches)
        # ``rows == offset`` means the drafter holds a context row for every
        # token the target has seen. A shortfall means priming restarted
        # part-way through the prompt and the drafter is working from a
        # suffix — correct, but blind to everything before the break.
        window = context_window()
        if window and rows >= window:
            note = f", windowed to {window}"
        elif rows >= history_offset:
            note = ""
        else:
            note = f", {history_offset - rows} missing"
        logger.info(
            "DSpark context ready: rows=%d offset=%d chunks=%d (primed%s)",
            rows,
            history_offset,
            ctx.chunks,
            note,
        )
        return ctx.caches, history_offset
    if isinstance(ctx, _PrimeCtx):
        logger.info(
            "DSpark priming dropped at the decode seam: context ends at %d, "
            "target cache at %d (expected %d) — %d row(s) lost",
            ctx.expected_target_offset,
            target_offset,
            target_offset - 1,
            _ctx_rows(ctx.caches),
        )
    seeded = host.make_dspark_cache()
    for cache in seeded:
        cache.seed_offset(target_offset - 1)
    logger.info(
        "DSpark context ready: rows=0 offset=%d (unprimed — drafting from the "
        "block alone until committed tokens accumulate)",
        target_offset - 1,
    )
    return seeded, target_offset - 1


# ---------------------------------------------------------------------------
# Registration helper (parity with deepseek_v4_dspark.register).
# ---------------------------------------------------------------------------


def register(namespace: Any) -> None:
    """Expose the DSpark classes on a model module, mlx-lm patch style."""
    if getattr(namespace, "DSparkHead", None) is not None:
        return
    namespace.DSparkContextCache = DSparkContextCache
    namespace.DSparkAttention = DSparkAttention
    namespace.DSparkDecoderLayer = DSparkDecoderLayer
    namespace.DSparkMarkovHead = DSparkMarkovHead
    namespace.DSparkConfidenceHead = DSparkConfidenceHead
    namespace.DSparkHead = DSparkHead


__all__ = [
    "CONFIG_KEYS",
    "DSparkArgs",
    "DSparkContextCache",
    "DSparkHead",
    "capture_prompt",
    "carry_config",
    "drop_primed",
    "dspark_args",
    "install_host_methods",
    "is_dspark_config",
    "max_draft_length",
    "register",
    "resolve_depth",
    "take_primed",
    "target_layer_ids",
]
