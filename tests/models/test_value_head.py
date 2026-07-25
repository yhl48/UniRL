"""CPU tests for ValueHead."""

from __future__ import annotations

import torch

from unirl.models.types.value_head import ValueHead


def test_value_head_output_shape() -> None:
    head = ValueHead(hidden_size=8)
    hidden = torch.randn(5, 8)
    values = head(hidden)
    assert values.shape == (5,)
    assert values.dtype == torch.float32
