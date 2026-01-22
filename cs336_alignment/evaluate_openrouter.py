"""
Evaluation script for models via OpenRouter API.
"""

from typing import Callable, List, Tuple
import os
import requests
from evaluation_lib import (
    load_system_prompt,
    load_math_parquet_data,
    save_evaluation_results,
    compute_evaluation_metrics,
    print_evaluation_summary,
    print_example_results,
)
from drgrpo_grader import r1_zero_reward_fn


def call_openrouter_api(
    prompt: str,
    api_key: str,
    model: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop: List[str] | None = None,
    add_think_tags: bool = False,
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
        add_think_tags: If True, wraps response with <think> tags before <answer>

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
    content = result["choices"][0]["message"]["content"]
    print(result)

    # Add think tags if requested
    if add_think_tags and "<answer>" in content:
        # Add <think> at the beginning and </think> before <answer>
        content = "<think>" + content.replace("<answer>", "</think> <answer>", 1)

    return content


def evaluate_openrouter(
    model: str,
    reward_fn: Callable[[str, str], dict[str, float]],
    data: List[Tuple[str, str]],
    system_prompt: str,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_tokens: int = 1024,
    stop: List[str] | None = None,
    output_path: str | None = None,
    api_key: str | None = None,
    data_limit: int | None = None,
    add_think_tags: bool = False,
):
    """
    Evaluate a language model via OpenRouter API on a dataset.

    Args:
        model: Model identifier (e.g., "openai/gpt-4")
        reward_fn: Function that takes (output, answer) and returns dict with 'format_reward', 'answer_reward', 'reward'
        data: List of (question, answer) tuples
        system_prompt: System prompt template with {question} placeholder
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        stop: List of stop sequences
        output_path: Optional path to save evaluation results as JSON
        api_key: OpenRouter API key (if None, reads from OPENROUTER_API_KEY environment variable)
        data_limit: Optional limit on number of examples to evaluate
        add_think_tags: If True, wraps response with <think> tags before <answer>

    Returns:
        Dictionary containing:
            - intermediate_results: List of dicts with prompts, answers, outputs, and scores
            - total_reward: Sum and accuracy for total reward
            - format_reward: Sum and accuracy for format reward
            - answer_reward: Sum and accuracy for answer reward
            - format_reward_when_answer_zero: Sum and accuracy for format reward when answer reward is 0
            - answer_reward_when_format_zero: Sum and accuracy for answer reward when format reward is 0
    """
    # Get API key from environment variable if not provided
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")
        if api_key is None:
            raise ValueError(
                "OpenRouter API key must be provided either as a parameter or "
                "via the OPENROUTER_API_KEY environment variable"
            )

    questions, answers = zip(*data)
    if data_limit:
        questions, answers = questions[:data_limit], answers[:data_limit]
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
            add_think_tags=add_think_tags,
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
    results = compute_evaluation_metrics(intermediate_results)

    # Print summary
    print_evaluation_summary(results)

    return results


# Example usage
if __name__ == "__main__":
    system_prompt = load_system_prompt("./cs336_alignment/prompts/deepseek_v3.prompt")
    # system_prompt = load_system_prompt("./cs336_alignment/prompts/r1_zero.prompt")

    # Using OpenRouter API
    # Set OPENROUTER_API_KEY environment variable before running
    results_openrouter = evaluate_openrouter(
        model="deepseek/deepseek-v3.2",
        # model="deepseek/deepseek-r1-0528",
        reward_fn=r1_zero_reward_fn,
        data=load_math_parquet_data(
            path="./data/MATH/data/test-00000-of-00001.parquet"
        ),
        system_prompt=system_prompt,
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        output_path="./evaluation_results_openrouter.json",
        data_limit=2,
        add_think_tags=True,  # Enable think tag wrapping
    )

    print_example_results(results_openrouter, 5)
