"""Shared AR replay result types."""

from __future__ import annotations

from typing import NamedTuple, Optional

import torch


class ARReplayOutput(NamedTuple):
    """Teacher-forced replay outputs for policy-gradient training."""

    log_probs: torch.Tensor
    values: Optional[torch.Tensor] = None


__all__ = ["ARReplayOutput"]
