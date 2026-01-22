"""
Evaluation script for models using vLLM.
"""
from typing import Callable, List, Tuple
from vllm import LLM, SamplingParams
from evaluation_lib import (
    load_system_prompt,
    load_math_parquet_data,
    save_evaluation_results,
    compute_evaluation_metrics,
    print_evaluation_summary,
    print_example_results,
)
from drgrpo_grader import r1_zero_reward_fn


def evaluate_vllm(
    model: LLM,
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[Tuple[str, str]],
    system_prompt: str,
    output_path: str | None = None,
):
    """
    Evaluate a language model on a dataset using vLLM.

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
    results = compute_evaluation_metrics(intermediate_results)

    # Print summary
    print_evaluation_summary(results)

    return results


# Example usage for vLLM
if __name__ == "__main__":
    system_prompt = load_system_prompt("./cs336_alignment/prompts/r1_zero.prompt")

    # Using vLLM (local)
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
