"""jang_tools' mlx_lm.models overrides must not outlive a JANG load.

jang_tools registers e.g. ``sys.modules["mlx_lm.models.deepseek_v4"] =
jang_tools.dsv4.mlx_model`` and never removes it, which broke every later
non-JANG DeepSeek-V4 load.
"""

import sys
import types

import pytest

from omlx.engine import jang
from omlx.engine.jang import _scoped_jang_model_modules, _serializes_requests

KEY = "mlx_lm.models._omni_scope_test"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(jang, "_JANG_MODEL_MODULES", {})
    monkeypatch.delitem(sys.modules, KEY, raising=False)
    yield
    sys.modules.pop(KEY, None)


def _module(name):
    return types.ModuleType(name)


def test_removes_jang_override_after_load():
    jang_mod = _module("jang_tools.fake.mlx_model")
    with _scoped_jang_model_modules():
        sys.modules[KEY] = jang_mod  # what jang_tools' register() does
    assert KEY not in sys.modules


def test_restores_the_module_that_was_there():
    upstream = _module("mlx_lm.models._omni_scope_test")
    sys.modules[KEY] = upstream
    with _scoped_jang_model_modules():
        sys.modules[KEY] = _module("jang_tools.fake.mlx_model")
    assert sys.modules[KEY] is upstream


def test_reinstalls_jang_override_for_later_loads():
    # register() runs only on first import, so the second load must get the
    # override back from the scope itself.
    jang_mod = _module("jang_tools.fake.mlx_model")
    upstream = _module("mlx_lm.models._omni_scope_test")
    with _scoped_jang_model_modules():
        sys.modules[KEY] = jang_mod
    sys.modules[KEY] = upstream
    with _scoped_jang_model_modules():
        assert sys.modules[KEY] is jang_mod
    assert sys.modules[KEY] is upstream


def test_leaves_non_jang_modules_alone():
    plain = _module("mlx_lm.models._omni_scope_test")
    with _scoped_jang_model_modules():
        sys.modules[KEY] = plain
    assert sys.modules[KEY] is plain


def test_serializes_jang_deepseek_v4_only():
    dsv4 = type("Model", (), {"__module__": "jang_tools.dsv4.mlx_model"})()
    other = type("Model", (), {"__module__": "mlx_lm.models.qwen3"})()
    wrapped = type("Adapter", (), {})()
    wrapped._model = dsv4
    assert _serializes_requests(dsv4)
    assert _serializes_requests(wrapped)
    assert not _serializes_requests(other)
