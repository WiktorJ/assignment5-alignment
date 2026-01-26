from dataclasses import dataclass
import re
import torch
from torch.autograd import grad
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from vllm import LLM, SamplingParams
from typing import Callable, Any, Tuple
from evaluate_model import evaluate_vllm
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import tyro
import json
import pandas as pd
import wandb
from torch.optim import AdamW
from drgrpo_grader import r1_zero_reward_fn


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    train_data_path: str = "data/MATH/data/evaluation_results_openrouter.jsonl"
    eval_data_path: str = "data/MATH/data/test-00000-of-00001.parquet"
    train_prompt_column: str = "prompt"
    train_response_column: str = "response"
    eval_prompt_column: str = "problem"
    eval_response_column: str = "answer"
    gpu_memory_utilization: float = 0.85
    dtype: str = "bfloat16"
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_length: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    learning_rate: float = 1e-5
    num_epochs: int = 1
    gradient_accumulation_steps: int = 1
    train_log_interval: int = 10
    eval_log_interval: int = 20


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    tokenized_prompts = [
        tokenizer.encode(prompt, return_tensors="pt").squeeze()
        for prompt in prompt_strs
    ]
    tokenized_outputs = [
        tokenizer.encode(
            output,
            return_tensors="pt",
        ).squeeze()
        for output in output_strs
    ]
    tokenized_inputs = [
        torch.cat((p, o), dim=0) for p, o in zip(tokenized_prompts, tokenized_outputs)
    ]
    max_len = max([len(t) for t in tokenized_inputs])
    labels = torch.full((len(tokenized_inputs), max_len), tokenizer.pad_token_id)
    input_ids = torch.full((len(tokenized_inputs), max_len), tokenizer.pad_token_id)

    for i, t in enumerate(tokenized_inputs):
        input_ids[i, : len(t)] = t
        labels[i, : len(t)] = t

    labels = labels[:, 1:]
    input_ids = input_ids[:, :-1]

    prompt_lens = torch.asarray([len(p) for p in tokenized_prompts])
    output_lens = torch.asarray([len(o) for o in tokenized_outputs])

    reponse_mask = (
        (torch.arange(1, labels.shape[1] + 1) >= prompt_lens[:, None])
        & (torch.arange(1, labels.shape[1] + 1) < (prompt_lens + output_lens)[:, None])
    ).int()
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": reponse_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy = -(probs * log_probs).sum(dim=-1)
    return entropy


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool,
) -> dict[str, torch.Tensor]:
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model.to(device)
    input_ids = input_ids.to(device)
    labels = labels.to(device)

    model.eval()
    with torch.inference_mode():
        outputs = model(input_ids)
        logits = outputs.logits
    log_probs = F.log_softmax(logits, dim=-1)  # B x SL x V
    response_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(
        -1
    )  # B x SL
    ret = {"log_probs": response_log_probs}
    if return_token_entropy:
        token_entropy = compute_entropy(logits)  # B x SL
        ret["token_entropy"] = token_entropy
    return ret


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
    normalize_constant: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float | None = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    def loss_fn(policy_log_probs, response_mask, normalize_constant):
        return -masked_normalize(
            tensor=policy_log_probs,
            mask=response_mask,
            normalize_constant=normalize_constant,
            dim=-1,
        )

    loss = loss_fn(policy_log_probs, response_mask, normalize_constant).mean() / (
        gradient_accumulation_steps
    )
    loss.backward()
    return loss, {}


def compute_eval_metrics(
    hf_model: PreTrainedModel | None,
    vllm_model: LLM,
    tokenizer: PreTrainedTokenizerBase,
    data: list[Tuple[str, str]],
    reward_fn: Callable[[str, str], dict[str, float]],
    batch_size: int,
    max_length: int,
    temperature: float,
    top_p: float,
) -> dict["str", Any]:
    """Log generations from a model to a file."""
    logs = []
    for i in range(0, len(data), batch_size):
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_length,
            stop=["</answer>"],
        )
        eval_results = evaluate_vllm(
            vllm_model, sampling_params, reward_fn, data[i : i + batch_size]
        )
        gt_answers = [res["answer"] for res in eval_results["intermediate_results"]]
        answers = [res["output"] for res in eval_results["intermediate_results"]]
        prompts = [res["prompt"] for res in eval_results["intermediate_results"]]
        format_rewards = [
            res["format_reward"] for res in eval_results["intermediate_results"]
        ]
        answer_rewards = [
            res["answer_reward"] for res in eval_results["intermediate_results"]
        ]
        total_rewards = [res["reward"] for res in eval_results["intermediate_results"]]

        # Only compute entropy/log_probs if HF model is provided
        if hf_model is not None:
            tokenized_output = tokenize_prompt_and_output(prompts, answers, tokenizer)
            response_output = get_response_log_probs(
                hf_model,
                tokenized_output["input_ids"],
                tokenized_output["labels"],
                return_token_entropy=True,
            )
            avg_entropy = masked_normalize(
                tensor=response_output["token_entropy"],
                mask=tokenized_output["response_mask"],
                normalize_constant=tokenized_output["response_mask"].sum(dim=-1),
                dim=-1,
            )
            avg_log_probs = masked_normalize(
                tensor=response_output["log_probs"],
                mask=tokenized_output["response_mask"],
                normalize_constant=tokenized_output["response_mask"].sum(dim=-1),
                dim=-1,
            )

        for j in range(len(prompts)):
            log_entry = {
                "prompt": prompts[j],
                "gt_answer": gt_answers[j],
                "answer": answers[j],
                "response_length": len(answers[j]),
                "format_reward": format_rewards[j],
                "answer_reward": answer_rewards[j],
                "total_reward": total_rewards[j],
            }
            if hf_model is not None:
                log_entry["entropy"] = avg_entropy[j].item()
                log_entry["log_probs"] = avg_log_probs[j].item()
            logs.append(log_entry)
    avg_response_len = torch.mean(
        torch.tensor([log["response_length"] for log in logs])
    )
    avg_correct_resp_len = torch.mean(
        torch.tensor(
            [log["response_length"] for log in logs if log["total_reward"] > 0]
        )
    )
    avg_incorrect_resp_len = torch.mean(
        torch.tensor(
            [log["response_length"] for log in logs if log["total_reward"] == 0]
        )
    )

    return {
        "full_logs": logs,
        "avg_response_len": avg_response_len.item(),
        "avg_correct_resp_len": avg_correct_resp_len.item(),
        "avg_incorrect_resp_len": avg_incorrect_resp_len.item(),
    }


def load_prompt_response_data_jsonl(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a JSONL file containing prompt and response fields.

    Args:
        file_path: Path to the JSONL file
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            data.append((item[prompt_column], item[response_column]))
    return data


def load_prompt_response_data_parquet(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a Parquet file containing prompt and response fields.

    Args:
        file_path: Path to the Parquet file
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)
    """

    df = pd.read_parquet(file_path)
    data = [(row[prompt_column], row[response_column]) for _, row in df.iterrows()]
    return data


def load_prompt_response_data(
    file_path: str, prompt_column: str = "prompt", response_column: str = "response"
) -> list[Tuple[str, str]]:
    """Load data from a file containing prompt and response fields.

    Supports both JSONL and Parquet file formats. The format is determined by the file extension.

    Args:
        file_path: Path to the data file (.jsonl or .parquet)
        prompt_column: Name of the column containing prompts
        response_column: Name of the column containing responses

    Returns:
        List of tuples where each tuple contains (prompt, response)

    Raises:
        ValueError: If the file extension is not supported
    """
    if file_path.endswith(".jsonl"):
        return load_prompt_response_data_jsonl(
            file_path, prompt_column, response_column
        )
    elif file_path.endswith(".parquet"):
        return load_prompt_response_data_parquet(
            file_path, prompt_column, response_column
        )
    else:
        raise ValueError(
            f"Unsupported file format. File must be .jsonl or .parquet, got: {file_path}"
        )


def init_vllm(
    model_id: str,
    device: str,
    dtype: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
):
    """Start the inference process, here we use vLLM to hold a model on a GPU separate from the policy. 13"""
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/ # 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=dtype,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel | dict, llm: LLM):
    """Copied from https://github.com/huggingface/trl/blob/ 22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670."""
    if isinstance(policy, PreTrainedModel):
        state_dict = policy.state_dict()
    else:
        state_dict = policy
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def train(config: TrainConfig):
    model_id = "Qwen/Qwen2.5-Math-1.5B"
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, config.dtype),
        device_map=config.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    train_data = load_prompt_response_data(
        config.train_data_path, config.train_prompt_column, config.train_response_column
    )
    eval_data = load_prompt_response_data(
        config.eval_data_path, config.eval_prompt_column, config.eval_response_column
    )

    wandb.init(mode="offline")
    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    # Setup optimizer
    optimizer = AdamW(hf_model.parameters(), lr=config.learning_rate)

    # Training loop
    hf_model.train()
    global_step = 0

    for epoch in range(config.num_epochs):
        train_data = [train_data[i] for i in torch.randperm(len(train_data))]
        for batch_idx in range(0, len(train_data), config.batch_size):
            # Get batch
            batch_data = train_data[batch_idx : batch_idx + config.batch_size]
            prompts = [item[0] for item in batch_data]
            responses = [item[1] for item in batch_data]

            tokenized_data = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized_data["input_ids"]
            labels = tokenized_data["labels"]
            response_mask = tokenized_data["response_mask"]

            probs = get_response_log_probs(
                hf_model, input_ids, labels, return_token_entropy=True
            )
            log_probs = probs["log_probs"]
            entropy = probs["token_entropy"]

            loss, train_metrics = sft_microbatch_train_step(
                log_probs,
                response_mask,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
            )
            if global_step % config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(hf_model.parameters(), max_norm=1.0)
                optimizer.zero_grad()
                optimizer.step()

            if (
                config.train_log_interval > 0
                and global_step % config.train_log_interval == 0
            ):
                avg_entropy = (
                    masked_normalize(
                        entropy,
                        response_mask,
                        dim=-1,
                        normalize_constant=response_mask.sum(dim=-1),
                    )
                    .mean()
                    .item()
                )
                wandb.log(
                    {
                        "train/avg_entropy": avg_entropy,
                        "train/avg_loss": loss.item()
                        * config.gradient_accumulation_steps,
                        "train/step": global_step,
                        "train/epoch": epoch,
                        **train_metrics,
                    }
                )
            if (
                config.eval_log_interval > 0
                and global_step % config.eval_log_interval == 0
            ):
                with torch.no_grad():
                    # Save model state and unload HF model to free GPU memory
                    print("Saving HF model state and unloading...")
                    model_state_dict = hf_model.state_dict()
                    del hf_model
                    torch.cuda.empty_cache()

                    # Load vLLM model for evaluation
                    print("Loading vLLM model for evaluation...")
                    vllm_model = init_vllm(
                        model_id,
                        device=config.device,
                        dtype=config.dtype,
                        seed=config.seed,
                        gpu_memory_utilization=config.gpu_memory_utilization,
                    )

                    # Load the trained weights into vLLM
                    load_policy_into_vllm_instance(model_state_dict, vllm_model)

                    eval_metrics = compute_eval_metrics(
                        hf_model=None,  # We don't need HF model for eval
                        vllm_model=vllm_model,
                        tokenizer=tokenizer,
                        data=eval_data,
                        reward_fn=r1_zero_reward_fn,
                        batch_size=config.batch_size,
                        max_length=config.max_length,
                        temperature=config.temperature,
                        top_p=config.top_p,
                    )
                    wandb.log(eval_metrics)

                    # Unload vLLM model
                    print("Unloading vLLM model...")
                    del vllm_model
                    torch.cuda.empty_cache()

                    # Reload HF model for continued training
                    print("Reloading HF model for training...")
                    hf_model = AutoModelForCausalLM.from_pretrained(
                        model_id,
                        torch_dtype=getattr(torch, config.dtype),
                        device_map=config.device,
                        trust_remote_code=True,
                    )
                    hf_model.load_state_dict(model_state_dict)
                    hf_model.train()
                    del model_state_dict
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    tyro.cli(train)
