"""Compiled module closures must die on the thread that built them.

jang_tools' DeepSeek-V4 decode path caches mx.compile'd closures over
``self`` in module __dict__s, so each trace holds the module's weights. The
JANGTQ warmup builds them on the loader thread and the engine thread builds
more on first decode; destroyed anywhere else, the traces were never released
and every unload after a generation leaked the whole model. JANGLoader drops
them on the loader thread after load and on the engine thread before close.
Whether a small model reproduces the leak varies from run to run, so this
pins the contract; the leak itself was verified on DeepSeek-V4-Flash-JANGTQ2
(70.9 GB retained before, 1.1 GB after, over two load/generate/unload
cycles).
"""

import mlx.core as mx
import mlx.nn as nn

from omlx.engine.jang import _drop_compiled_module_functions


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(2048, 2048, bias=False)  # 16 MB fp32
        self._note = "kept"

    def decode(self, x):
        fn = self.__dict__.get("_decode_compiled")
        if fn is None:
            fn = mx.compile(lambda xin: self.proj(xin))
            self.__dict__["_decode_compiled"] = fn
        return fn(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [_Block(), _Block()]
        mx.eval(self.parameters())

    def decode(self, x):
        for layer in self.layers:
            x = layer.decode(x)
        return x


def test_drops_only_compiled_functions():
    model = _Model()
    model.decode(mx.ones((1, 2048)))
    assert all("_decode_compiled" in vars(layer) for layer in model.layers)

    assert _drop_compiled_module_functions(model) == 2
    for layer in model.layers:
        assert "_decode_compiled" not in vars(layer)
        assert layer._note == "kept"
        assert "proj" in layer
