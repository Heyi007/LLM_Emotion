"""transformer 解码器层隐状态捕获工具。

Qwen3.8-27B 是 64 层混合骨干（线性注意力 + 每 4 层一次全注意力），
直接用 ``output_hidden_states=True`` 会一次性物化全部 65 个张量，显存开销大。
这里用 forward hook 只捕获指定层的残差流输出，离线抽取与在线推理共用同一套机制。
"""
from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """兼容不同包装类，定位 decoder layer 列表。"""
    candidates = [
        getattr(model, "model", None) and getattr(model.model, "layers", None),
        getattr(model, "language_model", None) and getattr(model.language_model, "layers", None),
        getattr(model, "layers", None),
    ]
    for layers in candidates:
        if layers is not None:
            return layers
    raise AttributeError("无法在模型上定位 decoder layers（.model.layers / .language_model.layers）")


class LayerCapture:
    """在指定 decoder 层注册 hook，捕获该层残差流输出 [B, L, H]。"""

    def __init__(self, model: nn.Module, layer_indexes: list[int]):
        self.layers = get_decoder_layers(model)
        self.layer_indexes = list(layer_indexes)
        self._handles = []
        self._buffer: dict[int, torch.Tensor] = {}

    def _make_hook(self, idx: int):
        def hook(_module, _args, output):
            # decoder 层输出通常是 (hidden_states, ...) 元组
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            self._buffer[idx] = hidden.detach()
        return hook

    def __enter__(self) -> "LayerCapture":
        for idx in self.layer_indexes:
            self._handles.append(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def __exit__(self, *exc) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get(self, idx: int) -> torch.Tensor:
        return self._buffer[idx]

    def clear(self) -> None:
        self._buffer.clear()


@contextmanager
def capture_layers(model: nn.Module, layer_indexes: list[int]):
    cap = LayerCapture(model, layer_indexes).__enter__()
    try:
        yield cap
    finally:
        cap.__exit__(None, None, None)
