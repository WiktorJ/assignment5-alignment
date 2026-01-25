import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from vllm import LLM, SamplingParams
from typing import Callable, Any, Tuple
from evaluate_model import evaluate_vllm


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


def stf_microbatch_train_step(
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


def log_generations(
    hf_model: PreTrainedModel,
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

        tokenized_output = tokenize_prompt_and_output(prompts, answers, tokenizer)
        avg_entropy = torch.zeros(
            tokenized_output["response_mask"].shape[0],
            device=tokenized_output["response_mask"].device,
        )
        avg_log_probs = torch.zeros(
            tokenized_output["response_mask"].shape[0],
            device=tokenized_output["response_mask"].device,
        )
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
            logs.append(
                {
                    "prompt": prompts[j],
                    "gt_answer": gt_answers[j],
                    "answer": answers[j],
                    "response_length": len(answers[j]),
                    "format_reward": format_rewards[j],
                    "answer_reward": answer_rewards[j],
                    "total_reward": total_rewards[j],
                    "entropy": avg_entropy[j].item(),
                    "log_probs": avg_log_probs[j].item(),
                }
            )
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
