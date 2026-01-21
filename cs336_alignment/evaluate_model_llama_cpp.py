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
    print_intermetiate_results: bool = False,
):
    questions, answers = zip(*data)
    # Interpolate each question into the system prompt
    prompts = [system_prompt.replace("{question}", question) for question in questions]
    outputs = model.generate(prompts, eval_sampling_params)
    scores = [
        reward_fn(output.outputs[0].text, answer)
        for output, answer in zip(outputs, answers)
    ]
    if print_intermetiate_results:
        for prompt, answer, output, score in zip(prompts, answers, outputs, scores):
            print(f"Prompt: {prompt}")
            print(f"Answer: {answer}")
            print(f"Output: {output.outputs[0].text}")
            print(f"Score: {score}")
            print("--------")

    total_format_reward = sum(score["format_reward"] for score in scores)
    total_answer_reward = sum(score["answer_reward"] for score in scores)
    total_reward = sum(score["reward"] for score in scores)
    print(f"Total format reward: {total_format_reward}")
    print(f"Total answer reward: {total_answer_reward}")
    print(f"Total reward: {total_reward}")
    print(f"Average format reward: {total_format_reward / len(scores)}")
    print(f"Average answer reward: {total_answer_reward / len(scores)}")
    print(f"Average reward: {total_reward / len(scores)}")

    return scores


# Example usage:
system_prompt = load_system_prompt("./cs336_alignment/prompts/r1_zero.prompt")
evaluate_llama(
    model=LLM(model="Qwen/Qwen2.5-Math-1.5B"),
    eval_sampling_params=SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
    ),
    reward_fn=r1_zero_reward_fn,
    data=load_math_parquet_data(path="./data/MATH/data/test-00000-of-00001.parquet"),
    system_prompt=system_prompt,
)
