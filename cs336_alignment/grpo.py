from typing import Callable
import torch


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_group_thruths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
):
    group_rewards = [
        reward_fn(response, truth)["reward"]
        for response, truth in zip(rollout_responses, repeated_group_thruths)
    ]
    group_rewards = torch.tensor(group_rewards).reshape(-1, group_size)
    group_mean_rewards = group_rewards.mean(dim=1)
    group_std_rewards = group_rewards.std(dim=1)
    advantages = group_rewards - group_mean_rewards[:, None]
    if normalize_by_std:
        advantages = advantages / (group_std_rewards[:, None] + advantage_eps)
    return (
        advantages,
        group_rewards,
        {
            "rewards_mean": group_rewards.mean().item(),
            "rewards_std": group_rewards.std().item(),
            "rewards_min": group_rewards.min().item(),
            "rewards_max": group_rewards.max().item(),
            "advantages_mean": advantages.mean().item(),
            "advantages_std": advantages.std().item(),
            "advantages_min": advantages.min().item(),
            "advantages_max": advantages.max().item(),
        },
    )
