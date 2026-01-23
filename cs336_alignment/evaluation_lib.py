"""
Common evaluation utilities shared across different evaluation methods.
"""
from typing import Callable, List, Tuple
import pandas as pd
from pathlib import Path
import json


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


def save_evaluation_results(
    intermediate_results: List[dict],
    output_path: str,
    append_only_last: bool = True,
) -> None:
    """
    Save evaluation results to a JSONL file (one JSON object per line).
    
    By default, only appends the last result to the file for efficiency.
    If the file doesn't exist or append_only_last is False, writes all results.

    Args:
        intermediate_results: List of result dictionaries containing prompts and outputs
        output_path: Path to save the JSONL file
        append_only_last: If True, only append the last result (efficient for incremental saves).
                         If False, rewrite entire file with all results.
    """
    from pathlib import Path
    
    file_exists = Path(output_path).exists()
    
    if append_only_last and file_exists and len(intermediate_results) > 0:
        # Append only the last result
        with open(output_path, "a") as f:
            result = intermediate_results[-1]
            output_item = {
                "prompt": result["prompt"],
                "response": result["output"],
            }
            f.write(json.dumps(output_item) + "\n")
    else:
        # Write all results (file doesn't exist or full rewrite requested)
        with open(output_path, "w") as f:
            for result in intermediate_results:
                output_item = {
                    "prompt": result["prompt"],
                    "response": result["output"],
                }
                f.write(json.dumps(output_item) + "\n")


def compute_evaluation_metrics(
    intermediate_results: List[dict],
) -> dict:
    """
    Compute all evaluation metrics from intermediate results.

    Args:
        intermediate_results: List of dicts with prompts, answers, outputs, and scores

    Returns:
        Dictionary containing all computed metrics
    """
    return {
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
