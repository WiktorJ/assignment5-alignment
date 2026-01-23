import torch


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
