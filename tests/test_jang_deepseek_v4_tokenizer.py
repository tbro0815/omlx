"""DeepSeek-V4 JANG checkpoints need the DSML chat template patch.

Their tokenizer_config.json has no chat template, so without the
deepseek_v4 tokenizer patch every chat request fails with
"tokenizer.chat_template is not set".
"""

import json

import pytest

from omlx.engine.jang import _install_deepseek_v4_tokenizer_patches


@pytest.fixture
def calls(monkeypatch):
    from omlx.patches.deepseek_v4 import tokenizer_patch

    seen = []
    monkeypatch.setattr(
        tokenizer_patch, "apply_tokenizer_patch", lambda: seen.append("tokenizer")
    )
    monkeypatch.setattr(tokenizer_patch, "apply_load_patch", lambda: seen.append("load"))
    return seen


def _model(tmp_path, model_type):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    return tmp_path


def test_installs_both_patches_for_deepseek_v4(tmp_path, calls):
    _install_deepseek_v4_tokenizer_patches(_model(tmp_path, "deepseek_v4"))
    assert calls == ["tokenizer", "load"]


@pytest.mark.parametrize("model_type", ["deepseek_v41", "qwen3_5", "deepseek_v3"])
def test_skips_other_model_types(tmp_path, calls, model_type):
    _install_deepseek_v4_tokenizer_patches(_model(tmp_path, model_type))
    assert calls == []


def test_skips_unreadable_config(tmp_path, calls):
    _install_deepseek_v4_tokenizer_patches(tmp_path)
    (tmp_path / "config.json").write_text("{not json")
    _install_deepseek_v4_tokenizer_patches(tmp_path)
    assert calls == []
