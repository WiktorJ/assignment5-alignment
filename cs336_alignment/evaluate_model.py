from typing import Callable, List, Tuple
import pandas as pd
from pathlib import Path
import json
import requests
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn


def load_system_prompt(prompt_path: str) -> str:
    """
    Load system prompt from a file.

    Args:
        prompt_path: Path to the prompt file

    Returns:
        The system prompt as a string
    """
    with open(prompt_path, "r") as f:
        return f.read()


def calculate_reward_metrics(intermediate_results: List[dict], reward_key: str) -> dict:
    """
    Calculate sum and accuracy metrics for a specific reward type.

    Args:
        intermediate_results: List of result dictionaries
        reward_key: Key to extract from results ('reward', 'format_reward', or 'answer_reward')

    Returns:
        Dictionary with 'sum' and 'accuracy' keys
    """
    n = len(intermediate_results)
    reward_sum = sum(r[reward_key] for r in intermediate_results)
    accuracy = sum(1 for r in intermediate_results if r[reward_key] > 0) / n
    return {"sum": reward_sum, "accuracy": accuracy}


def calculate_conditional_reward_metrics(
    intermediate_results: List[dict],
    condition_key: str,
    condition_value: float,
    reward_key: str,
) -> dict:
    """
    Calculate reward metrics when another reward meets a condition.

    Args:
        intermediate_results: List of result dictionaries
        condition_key: Key to check condition on
        condition_value: Value to compare against
        reward_key: Key to calculate metrics for

    Returns:
        Dictionary with 'sum', 'accuracy', and 'count' keys
    """
    filtered_results = [
        r for r in intermediate_results if r[condition_key] == condition_value
    ]

    if filtered_results:
        reward_sum = sum(r[reward_key] for r in filtered_results)
        accuracy = sum(1 for r in filtered_results if r[reward_key] > 0) / len(
            filtered_results
        )
    else:
        reward_sum = 0
        accuracy = 0

    return {"sum": reward_sum, "accuracy": accuracy, "count": len(filtered_results)}


def print_evaluation_summary(results: dict) -> None:
    """
    Print a formatted summary of evaluation results.

    Args:
        results: Dictionary containing evaluation metrics
    """
    n = len(results["intermediate_results"])
    print(f"Total samples: {n}")

    print(
        f"\nTotal Reward - Sum: {results['total_reward']['sum']:.2f}, "
        f"Accuracy: {results['total_reward']['accuracy']:.2%}"
    )
    print(
        f"Format Reward - Sum: {results['format_reward']['sum']:.2f}, "
        f"Accuracy: {results['format_reward']['accuracy']:.2%}"
    )
    print(
        f"Answer Reward - Sum: {results['answer_reward']['sum']:.2f}, "
        f"Accuracy: {results['answer_reward']['accuracy']:.2%}"
    )

    print(
        f"\nFormat Reward when Answer=0 - "
        f"Sum: {results['format_reward_when_answer_zero']['sum']:.2f}, "
        f"Accuracy: {results['format_reward_when_answer_zero']['accuracy']:.2%}, "
        f"Count: {results['format_reward_when_answer_zero']['count']}"
    )
    print(
        f"Answer Reward when Format=0 - "
        f"Sum: {results['answer_reward_when_format_zero']['sum']:.2f}, "
        f"Accuracy: {results['answer_reward_when_format_zero']['accuracy']:.2%}, "
        f"Count: {results['answer_reward_when_format_zero']['count']}"
    )


def print_example_results(
    results: dict,
    n_examples: int = 3,
    groups: List[str] | None = None,
) -> None:
    """
    Print well-formatted examples from different groups of intermediate results.

    Args:
        results: Dictionary containing evaluation metrics and intermediate results
        n_examples: Number of examples to print per group
        groups: List of groups to print. Options:
            - 'both_correct': Both format and answer rewards are positive
            - 'format_only': Format correct but answer wrong
            - 'answer_only': Answer correct but format wrong
            - 'both_wrong': Both format and answer rewards are zero
            - 'all': Print examples from all groups
            If None, defaults to ['all']
    """
    if groups is None:
        groups = ["all"]

    intermediate_results = results["intermediate_results"]

    # Define group filters
    group_filters = {
        "both_correct": lambda r: r["format_reward"] > 0 and r["answer_reward"] > 0,
        "format_only": lambda r: r["format_reward"] > 0 and r["answer_reward"] == 0,
        "answer_only": lambda r: r["format_reward"] == 0 and r["answer_reward"] > 0,
        "both_wrong": lambda r: r["format_reward"] == 0 and r["answer_reward"] == 0,
    }

    # If 'all' is specified, print from all groups
    if "all" in groups:
        groups = list(group_filters.keys())

    print("\n" + "=" * 80)
    print("EXAMPLE RESULTS BY GROUP")
    print("=" * 80)

    for group_name in groups:
        if group_name not in group_filters:
            print(f"\nWarning: Unknown group '{group_name}', skipping...")
            continue

        # Filter results for this group
        group_results = [
            r for r in intermediate_results if group_filters[group_name](r)
        ]

        print(f"\n{'─' * 80}")
        print(f"GROUP: {group_name.upper().replace('_', ' ')}")
        print(f"Total in group: {len(group_results)}")
        print(f"{'─' * 80}")

        if not group_results:
            print("  No examples in this group.")
            continue

        # Print up to n_examples from this group
        for i, result in enumerate(group_results[:n_examples]):
            print(f"\n  Example {i + 1}/{min(n_examples, len(group_results))}:")
            print(f"  {'-' * 76}")
            print(f"  Prompt (first 200 chars):")
            print(f"    {result['prompt'][:200]}...")
            print(f"\n  Expected Answer:")
            print(f"    {result['answer']}")
            print(f"\n  Model Output:")
            print(f"    {result['output']}")
            print(f"\n  Rewards:")
            print(f"    Format: {result['format_reward']:.2f}")
            print(f"    Answer: {result['answer_reward']:.2f}")
            print(f"    Total:  {result['reward']:.2f}")

        if len(group_results) > n_examples:
            print(
                f"\n  ... and {len(group_results) - n_examples} more examples in this group"
            )

    print("\n" + "=" * 80)


def load_math_parquet_data(path: str = "./data/MATH/data") -> List[tuple[str, str]]:
    """
    Load MATH dataset from parquet file(s).

    Args:
        path: Path to a specific parquet file or directory containing parquet files
              with 'problem' and 'answer' columns

    Returns:
        List of (problem, answer) tuples
    """
    data_path = Path(path)
    results = []

    # Determine if path is a file or directory
    if data_path.is_file():
        if not data_path.suffix == ".parquet":
            raise ValueError(f"File {path} is not a parquet file")
        parquet_files = [data_path]
    elif data_path.is_dir():
        # Find all parquet files in the directory
        parquet_files = list(data_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {path}")
    else:
        raise FileNotFoundError(f"Path {path} does not exist")

    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)

        # Verify required columns exist
        if "problem" not in df.columns or "answer" not in df.columns:
            raise ValueError(
                f"File {parquet_file} missing 'problem' or 'answer' column"
            )

        # Convert to list of tuples
        for _, row in df.iterrows():
            results.append((str(row["problem"]), str(row["answer"])))

    return results


def call_openrouter_api(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop: List[str] | None = None,
) -> str:
    """
    Call OpenRouter API endpoint to generate a response.

    Args:
        prompt: The prompt to send to the model
        api_key: OpenRouter API key
        model: Model identifier (e.g., "openai/gpt-4")
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        stop: List of stop sequences

    Returns:
        The generated text response
    """
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

    if stop:
        payload["stop"] = stop

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    result = response.json()
    return result["choices"][0]["message"]["content"]


def save_evaluation_results(
    intermediate_results: List[dict],
    output_path: str,
) -> None:
    """
    Save evaluation results to a JSON file.

    Args:
        intermediate_results: List of result dictionaries containing prompts and outputs
        output_path: Path to save the JSON file
    """
    # Extract only prompt and response for each result
    output_data = [
        {
            "prompt": result["prompt"],
            "response": result["output"],
        }
        for result in intermediate_results
    ]

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(output_data)} evaluation results to {output_path}")


def evaluate_openrouter(
    api_key: str,
    model: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[Tuple[str, str]],
    system_prompt: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop: List[str] | None = None,
    output_path: str | None = None,
):
    """
    Evaluate a language model via OpenRouter API on a dataset.

    Args:
        api_key: OpenRouter API key
        model: Model identifier (e.g., "openai/gpt-4")
        reward_fn: Function that takes (output, answer) and returns dict with 'format_reward', 'answer_reward', 'reward'
        data: List of (question, answer) tuples
        system_prompt: System prompt template with {question} placeholder
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        stop: List of stop sequences
        output_path: Optional path to save evaluation results as JSON

    Returns:
        Dictionary containing:
            - intermediate_results: List of dicts with prompts, answers, outputs, and scores
            - total_reward: Sum and accuracy for total reward
            - format_reward: Sum and accuracy for format reward
            - answer_reward: Sum and accuracy for answer reward
            - format_reward_when_answer_zero: Sum and accuracy for format reward when answer reward is 0
            - answer_reward_when_format_zero: Sum and accuracy for answer reward when format reward is 0
    """
    questions, answers = zip(*data)
    # Interpolate each question into the system prompt
    prompts = [system_prompt.replace("{question}", question) for question in questions]

    # Collect intermediate results
    intermediate_results = []
    for i, (prompt, answer) in enumerate(zip(prompts, answers)):
        print(f"Processing {i + 1}/{len(prompts)}...", end="\r")

        # Call OpenRouter API
        output_text = call_openrouter_api(
            prompt=prompt,
            api_key=api_key,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
        )

        # Calculate reward
        score = reward_fn(output_text, answer)
        intermediate_results.append(
            {
                "prompt": prompt,
                "answer": answer,
                "output": output_text,
                "format_reward": score["format_reward"],
                "answer_reward": score["answer_reward"],
                "reward": score["reward"],
            }
        )

    print()  # New line after progress indicator

    # Save results if output path is provided
    if output_path:
        save_evaluation_results(intermediate_results, output_path)

    # Calculate metrics using helper functions
    results = {
        "intermediate_results": intermediate_results,
        "total_reward": calculate_reward_metrics(intermediate_results, "reward"),
        "format_reward": calculate_reward_metrics(
            intermediate_results, "format_reward"
        ),
        "answer_reward": calculate_reward_metrics(
            intermediate_results, "answer_reward"
        ),
        "format_reward_when_answer_zero": calculate_conditional_reward_metrics(
            intermediate_results, "answer_reward", 0, "format_reward"
        ),
        "answer_reward_when_format_zero": calculate_conditional_reward_metrics(
            intermediate_results, "format_reward", 0, "answer_reward"
        ),
    }

    # Print summary
    print_evaluation_summary(results)

    return results


def evaluate_vllm(
    model: LLM,
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[Tuple[str, str]],
    system_prompt: str,
    output_path: str | None = None,
):
    """
    Evaluate a language model on a dataset.

    Args:
        model: The LLM model to evaluate
        eval_sampling_params: Sampling parameters for generation
        reward_fn: Function that takes (output, answer) and returns dict with 'format_reward', 'answer_reward', 'reward'
        data: List of (question, answer) tuples
        system_prompt: System prompt template with {question} placeholder
        output_path: Optional path to save evaluation results as JSON

    Returns:
        Dictionary containing:
            - intermediate_results: List of dicts with prompts, answers, outputs, and scores
            - total_reward: Sum and accuracy for total reward
            - format_reward: Sum and accuracy for format reward
            - answer_reward: Sum and accuracy for answer reward
            - format_reward_when_answer_zero: Sum and accuracy for format reward when answer reward is 0
            - answer_reward_when_format_zero: Sum and accuracy for answer reward when format reward is 0
    """
    questions, answers = zip(*data)
    # Interpolate each question into the system prompt
    prompts = [system_prompt.replace("{question}", question) for question in questions]
    outputs = model.generate(prompts, eval_sampling_params)

    # Collect intermediate results
    intermediate_results = []
    for prompt, answer, output in zip(prompts, answers, outputs):
        score = reward_fn(output.outputs[0].text, answer)
        intermediate_results.append(
            {
                "prompt": prompt,
                "answer": answer,
                "output": output.outputs[0].text,
                "format_reward": score["format_reward"],
                "answer_reward": score["answer_reward"],
                "reward": score["reward"],
            }
        )

    # Save results if output path is provided
    if output_path:
        save_evaluation_results(intermediate_results, output_path)

    # Calculate metrics using helper functions
    results = {
        "intermediate_results": intermediate_results,
        "total_reward": calculate_reward_metrics(intermediate_results, "reward"),
        "format_reward": calculate_reward_metrics(
            intermediate_results, "format_reward"
        ),
        "answer_reward": calculate_reward_metrics(
            intermediate_results, "answer_reward"
        ),
        "format_reward_when_answer_zero": calculate_conditional_reward_metrics(
            intermediate_results, "answer_reward", 0, "format_reward"
        ),
        "answer_reward_when_format_zero": calculate_conditional_reward_metrics(
            intermediate_results, "format_reward", 0, "answer_reward"
        ),
    }

    # Print summary
    print_evaluation_summary(results)

    return results


# Example usage for vLLM:
if __name__ == "__main__":
    system_prompt = load_system_prompt("./cs336_alignment/prompts/r1_zero.prompt")

    # Example 1: Using vLLM (local)
    results = evaluate_vllm(
        model=LLM(model="Qwen/Qwen2.5-Math-1.5B"),
        eval_sampling_params=SamplingParams(
            temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
        ),
        reward_fn=r1_zero_reward_fn,
        data=load_math_parquet_data(
            path="./data/MATH/data/test-00000-of-00001.parquet"
        ),
        system_prompt=system_prompt,
        output_path="./evaluation_results_vllm.json",
    )
    print_example_results(results, 10)

    # Example 2: Using OpenRouter API
    # Uncomment and set your API key to use
    # results_openrouter = evaluate_openrouter(
    #     api_key="YOUR_OPENROUTER_API_KEY",
    #     model="openai/gpt-4",
    #     reward_fn=r1_zero_reward_fn,
    #     data=load_math_parquet_data(path="./data/MATH/data/test-00000-of-00001.parquet"),
    #     system_prompt=system_prompt,
    #     temperature=1.0,
    #     top_p=1.0,
    #     max_tokens=1024,
    #     stop=["\n"],
    #     output_path="./evaluation_results_openrouter.json",
    # )
    # print_example_results(results_openrouter, 10)
