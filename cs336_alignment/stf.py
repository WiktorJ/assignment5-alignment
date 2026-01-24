import torch
import torch.nn.functional as F
from transformers import PreTrainedModel


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
