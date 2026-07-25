"""CPU tests for scatter_terminal_rewards and track-level GAE wiring."""

from __future__ import annotations

import torch

from unirl.types.advantages import scatter_terminal_rewards
from unirl.types.rollout_resp import RolloutTrack
from unirl.types.segments.text import TextSegment


def test_scatter_terminal_rewards_places_reward_on_last_token() -> None:
    segment = TextSegment.pack(
        tokens=[torch.tensor([10, 11]), torch.tensor([20])],
        values=[torch.tensor([0.2, 0.5]), torch.tensor([0.8])],
    )
    assert segment.cu_seqlens is not None
    assert segment.lengths is not None
    rewards = torch.tensor([1.0, 0.5])
    token_rewards = scatter_terminal_rewards(
        rewards, lengths=segment.lengths, cu_seqlens=segment.cu_seqlens
    )
    assert token_rewards.shape == (3,)
    assert token_rewards.tolist() == [0.0, 1.0, 0.5]


def test_rollout_track_compute_gae_advantages() -> None:
    segment = TextSegment.pack(
        tokens=[torch.tensor([10, 11, 12])],
        values=[torch.tensor([0.2, 0.5, 0.8])],
    )
    track = RolloutTrack(
        sample_ids=["s0"],
        rewards=torch.tensor([1.0]),
        segment=segment,
    )
    updated = track.compute_gae_advantages(gamma=1.0, gae_lambda=1.0)
    assert updated.segment is not None
    assert updated.segment.token_advantages is not None
    assert updated.segment.returns is not None
    assert updated.segment.token_advantages.shape == (3,)
    assert updated.advantages is not None
    assert updated.advantages.shape == (1,)
    # Hand-check from PR1 test: sparse terminal reward on 3 tokens.
    expected = torch.tensor([0.8, 0.5, 0.2])
    assert torch.allclose(updated.segment.token_advantages, expected, atol=1e-6)


def test_rollout_track_compute_gae_advantages_multi_sample_no_leak() -> None:
    """Packed batch: GAE must not bootstrap across trajectory boundaries."""
    segment = TextSegment.pack(
        tokens=[torch.tensor([10, 11]), torch.tensor([20])],
        values=[torch.tensor([0.2, 0.5]), torch.tensor([0.8])],
    )
    track = RolloutTrack(
        sample_ids=["s0", "s1"],
        rewards=torch.tensor([1.0, 0.5]),
        segment=segment,
    )
    updated = track.compute_gae_advantages(gamma=1.0, gae_lambda=1.0)
    assert updated.segment is not None
    assert updated.segment.token_advantages is not None

    track0 = RolloutTrack(
        sample_ids=["s0"],
        rewards=torch.tensor([1.0]),
        segment=TextSegment.pack(
            tokens=[torch.tensor([10, 11])],
            values=[torch.tensor([0.2, 0.5])],
        ),
    )
    expected0 = track0.compute_gae_advantages(gamma=1.0, gae_lambda=1.0).segment
    assert expected0 is not None and expected0.token_advantages is not None

    track1 = RolloutTrack(
        sample_ids=["s1"],
        rewards=torch.tensor([0.5]),
        segment=TextSegment.pack(
            tokens=[torch.tensor([20])],
            values=[torch.tensor([0.8])],
        ),
    )
    expected1 = track1.compute_gae_advantages(gamma=1.0, gae_lambda=1.0).segment
    assert expected1 is not None and expected1.token_advantages is not None

    packed_adv = updated.segment.token_advantages
    assert torch.allclose(packed_adv[:2], expected0.token_advantages, atol=1e-6)
    assert torch.allclose(packed_adv[2:], expected1.token_advantages, atol=1e-6)
