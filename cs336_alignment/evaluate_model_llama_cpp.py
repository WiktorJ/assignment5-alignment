from typing import Callable, List, Tuple
import pandas as pd
from pathlib import Path
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


def calculate_reward_metrics(
    intermediate_results: List[dict], reward_key: str
) -> dict:
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


def evaluate_llama(
    model: LLM,
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[Tuple[str, str]],
    system_prompt: str,
):
    """
    Evaluate a language model on a dataset.

    Args:
        model: The LLM model to evaluate
        eval_sampling_params: Sampling parameters for generation
        reward_fn: Function that takes (output, answer) and returns dict with 'format_reward', 'answer_reward', 'reward'
        data: List of (question, answer) tuples
        system_prompt: System prompt template with {question} placeholder
        print_intermediate_results: Whether to print intermediate results

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

    # Calculate metrics using helper functions
    results = {
        "intermediate_results": intermediate_results,
        "total_reward": calculate_reward_metrics(intermediate_results, "reward"),
        "format_reward": calculate_reward_metrics(intermediate_results, "format_reward"),
        "answer_reward": calculate_reward_metrics(intermediate_results, "answer_reward"),
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


# Example usage:
# system_prompt = load_system_prompt("./cs336_alignment/prompts/r1_zero.prompt")
# results = evaluate_llama(
#     model=LLM(model="Qwen/Qwen2.5-Math-1.5B"),
#     eval_sampling_params=SamplingParams(
#         temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
#     ),
#     reward_fn=r1_zero_reward_fn,
#     data=load_math_parquet_data(path="./data/MATH/data/test-00000-of-00001.parquet"),
#     system_prompt=system_prompt,
# )
