from typing import Callable, Literal, Any
import torch
import wandb
from dataclasses import dataclass
import tyro
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from vllm import LLM, SamplingParams
from cs336_alignment.evaluate_model import evaluate_vllm
from cs336_alignment.file_util import load_prompt_response_data
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.stf import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    masked_normalize,
)
from torch.optim import AdamW
import bitsandbytes as bnb


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    train_data_path: str = (
        "data/MATH/data/deepseek_generated_train_data_max_len_1024_2048.jsonl"
    )
    eval_data_path: str = "data/MATH/data/test-00000-of-00001.parquet"
    train_prompt_column: str = "prompt"
    train_response_column: str = "response"
    eval_prompt_column: str = "problem"
    eval_response_column: str = "answer"
    gpu_memory_utilization: float = 0.9
    dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"
    use_8bit_adamw: bool = True
    train_log_interval: int = 10
    eval_log_interval: int = 0
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    advantage_eps: float = 1e-6
    clip_range: float = 1.0
    rollout_batch_size: int = 256
    group_size: int = 8
    temperature: float = 1.0
    top_p: float = 1.0
    min_length: int = 4  # As in Expiter, disallow empty string responses
    max_length: int = 1024
    epochs_per_rollout_batch: int = 1  # On-policy
    train_batch_size: int = 256  # On-policy
    gradient_accumulation_steps: int = 128  # microbatch size is 2, will fit on H100
    gpu_memory_utilization: float = 0.85
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
    ] = "reinforce_with_baseline"
    use_std_normalization: bool = True


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_group_thruths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
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
        advantages.reshape(-1),
        group_rewards.reshape(-1),
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


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -policy_log_probs * raw_rewards_or_advantages.squeeze()[:, None]


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratio = torch.exp(policy_log_probs - old_log_probs)
    non_clipped_loss = compute_naive_policy_gradient_loss(advantages, ratio)
    clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
    clipped_loss = compute_naive_policy_gradient_loss(advantages, clipped_ratio)
    return torch.max(
        compute_naive_policy_gradient_loss(advantages, ratio),
        compute_naive_policy_gradient_loss(
            advantages, torch.clamp(ratio, 1 - cliprange, 1 + cliprange)
        ),
    ), {
        "ratios": ratio,
        "non_clipped_loss": non_clipped_loss,
        "clipped_loss": clipped_loss,
        "clipped_or_not": clipped_ratio == ratio,
        "policy_log_probs": policy_log_probs,
        "old_log_probs": old_log_probs,
    }


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def no_baseline_loss(raw_rewards: torch.Tensor):
        return compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)

    def reinforce_with_baseline_loss(advantages: torch.Tensor):
        return compute_naive_policy_gradient_loss(advantages, policy_log_probs)

    def grpo_clip_loss(
        advantages: torch.Tensor, old_log_probs: torch.Tensor, cliprange: float
    ):
        return compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )

    if loss_type == "no_baseline":
        assert raw_rewards is not None
        return no_baseline_loss(raw_rewards), {}
    if loss_type == "reinforce_with_baseline":
        assert advantages is not None
        return reinforce_with_baseline_loss(advantages), {}
    if loss_type == "grpo_clip":
        assert advantages is not None
        assert old_log_probs is not None
        assert cliprange is not None
        return grpo_clip_loss(advantages, old_log_probs, cliprange)
    raise ValueError(f"Unknown loss type: {loss_type}")


def masked_mean(
    tensor: torch.Tensor, mask: torch.Tensor, dim: int | None = None
) -> torch.Tensor:
    return torch.sum(tensor * mask, dim=dim) / torch.sum(mask, dim=dim)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    policy_log_probs = policy_log_probs.to(device)
    response_mask = response_mask.to(device)
    if raw_rewards is not None:
        raw_rewards = raw_rewards.to(device)
    if advantages is not None:
        advantages = advantages.to(device)
    if old_log_probs is not None:
        old_log_probs = old_log_probs.to(device)
    token_loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    loss = masked_mean(token_loss, response_mask, dim=-1).mean()
    loss = loss / gradient_accumulation_steps
    loss.backward()
    return loss, metadata


def sample_rollout(
    vllm_model: LLM,
    data: list[tuple[str, str]],
    group_size: int,
    reward_fn: Callable[[str, str], dict[str, float]],
    max_length: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    sampling_params = SamplingParams(
        max_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
        stop=["</answer>"],
        n=group_size,
    )
    rollouts = evaluate_vllm(
        model=vllm_model,
        eval_sampling_params=sampling_params,
        data=data,
        reward_fn=reward_fn,
    )["intermediate_results"]
    return {
        "promtps": [r["prompt"] for r in rollouts],
        "responses": [r["output"] for r in rollouts],
        "answers": [r["answer"] for r in rollouts],
        "format_rewards": [r["format_reward"] for r in rollouts],
        "answer_rewards": [r["answer_reward"] for r in rollouts],
        "rewards": [r["reward"] for r in rollouts],
    }


def train(config: TrainConfig):
    assert config.train_batch_size >= config.rollout_batch_size
    assert config.train_batch_size % config.rollout_batch_size == 0

    print("=" * 80)
    print("Starting GRPO Training")
    print("=" * 80)
    print(f"Model: Qwen/Qwen2.5-Math-1.5B")
    print(f"Total GRPO steps: {config.n_grpo_steps}")
    print(f"Rollout batch size: {config.rollout_batch_size}")
    print(f"Group size: {config.group_size}")
    print(f"Train batch size: {config.train_batch_size}")
    print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Loss type: {config.loss_type}")
    print("=" * 80)

    model_id = "Qwen/Qwen2.5-Math-1.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, config.dtype),
        device_map=config.device,
        trust_remote_code=True,
        attn_implementation=config.attn_implementation,
    )
    hf_model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    train_data = load_prompt_response_data(
        config.train_data_path, config.train_prompt_column, config.train_response_column
    )
    eval_data = load_prompt_response_data(
        config.eval_data_path, config.eval_prompt_column, config.eval_response_column
    )

    wandb.init(mode="offline", sync_tensorboard=True)
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Setup optimizer
    if config.use_8bit_adamw:
        optimizer = bnb.optim.AdamW8bit(hf_model.parameters(), lr=config.learning_rate)
    else:
        optimizer = AdamW(hf_model.parameters(), lr=config.learning_rate)

    # Training loop
    hf_model.train()
    n_prompts_per_rollout = config.rollout_batch_size // config.group_size
    micro_batch_size = config.train_batch_size // config.gradient_accumulation_steps
    global_step = 0
    
    print(f"\nStarting training loop with {len(train_data)} training examples")
    print(f"Prompts per rollout: {n_prompts_per_rollout}")
    print(f"Micro batch size: {micro_batch_size}\n")
    
    for grpo_step in range(config.n_grpo_steps):
        print(f"\n{'='*80}")
        print(f"GRPO Step {grpo_step + 1}/{config.n_grpo_steps}")
        print(f"{'='*80}")
        
        for rollout_idx in range(0, len(train_data), n_prompts_per_rollout):
            print(f"\nRollout {rollout_idx // n_prompts_per_rollout + 1} - Processing prompts {rollout_idx} to {rollout_idx + n_prompts_per_rollout}")
            rollout_train_data = train_data[
                rollout_idx : rollout_idx + config.rollout_batch_size
            ]
            
            print(f"  Sampling {config.group_size} rollouts per prompt...")
            rollouts = sample_rollout(
                vllm_model=hf_model,
                data=rollout_train_data,
                group_size=config.group_size,
                reward_fn=r1_zero_reward_fn,
                max_length=config.max_length,
                temperature=config.temperature,
                top_p=config.top_p,
            )
            repeated_group_thruths = [
                answer
                for answer in rollouts["answers"]
                for _ in range(config.group_size)
            ]
            rolledout_responses = [
                r for r in rollouts["responses"] for _ in range(config.group_size)
            ]
            rolled_out_prompts = [
                r for r in rollouts["promtps"] for _ in range(config.group_size)
            ]
            advantages, group_rewards, metadata = compute_group_normalized_rewards(
                reward_fn=r1_zero_reward_fn,
                rollout_responses=rolledout_responses,
                repeated_group_thruths=repeated_group_thruths,
                group_size=config.group_size,
                advantage_eps=config.advantage_eps,
                normalize_by_std=config.use_std_normalization,
            )
            
            print(f"  Computed rewards - Mean: {metadata['rewards_mean']:.4f}, Std: {metadata['rewards_std']:.4f}")
            print(f"  Advantages - Mean: {metadata['advantages_mean']:.4f}, Std: {metadata['advantages_std']:.4f}")
            tokenized_data = tokenize_prompt_and_output(
                rolled_out_prompts, rolledout_responses, tokenizer
            )
            input_ids = tokenized_data["input_ids"]
            labels = tokenized_data["labels"]
            response_mask = tokenized_data["response_mask"]
            old_log_probs = torch.empty()
            for batch_idx in range(0, len(advantages), micro_batch_size):
                batch_input_ids = input_ids[
                    batch_idx : batch_idx + config.train_batch_size
                ]
                batch_labels = labels[batch_idx : batch_idx + config.train_batch_size]

                old_log_probs[batch_idx : batch_idx + config.train_batch_size] = (
                    get_response_log_probs(
                        hf_model,
                        batch_input_ids,
                        batch_labels,
                        return_token_entropy=False,
                        inference_mode=True,
                    )["log_probs"]
                )

            for epoch in range(config.epochs_per_rollout_batch):
                print(f"  Epoch {epoch + 1}/{config.epochs_per_rollout_batch}")
                
                for batch_idx in range(0, len(advantages), micro_batch_size):
                    batch_advantages = advantages[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    batch_group_rewards = group_rewards[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    batch_input_ids = input_ids[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    batch_labels = labels[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    batch_response_mask = response_mask[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    batch_old_log_probs = old_log_probs[
                        batch_idx : batch_idx + config.train_batch_size
                    ]
                    probs = get_response_log_probs(
                        hf_model,
                        batch_input_ids,
                        batch_labels,
                        return_token_entropy=True,
                        inference_mode=False,
                    )
                    loss, loss_metadata = grpo_microbatch_train_step(
                        probs["log_probs"],
                        batch_response_mask,
                        config.gradient_accumulation_steps,
                        config.loss_type,
                        batch_group_rewards,
                        batch_advantages,
                        batch_old_log_probs,
                        config.clip_range,
                    )

                    if global_step % config.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        print(f"    Optimizer step at global_step {global_step}")
                    global_step += 1
                    if global_step % config.train_log_interval == 0:
                        avg_entropy = (
                            masked_normalize(
                                probs["entropy"],
                                response_mask,
                                dim=-1,
                                normalize_constant=response_mask.sum(dim=-1),
                            )
                            .mean()
                            .item()
                        )
                        avg_loss = loss.item() * config.gradient_accumulation_steps
                        print(
                            f"\n[Train Log] Step {global_step} | Loss: {avg_loss:.4f} | Entropy: {avg_entropy:.4f}"
                        )
                        wandb.log(
                            {
                                "train/avg_entropy": avg_entropy,
                                "train/avg_loss": avg_loss,
                                "train/step": global_step,
                                "train/epoch": epoch,
                                **loss_metadata,
                            }
                        )


if __name__ == "__main__":
    tyro.cli(train)
