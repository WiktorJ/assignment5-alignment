"""
Evaluation script for models via OpenRouter API.
"""

from typing import Callable, List, Tuple
import os
import time
import json
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
    check_reasoning: bool = False,
    max_retries: int = 3,
) -> str:
    """
    Call OpenRouter API endpoint to generate a response with retry logic.

    Args:
        prompt: The prompt to send to the model
        api_key: OpenRouter API key
        model: Model identifier (e.g., "openai/gpt-4")
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        stop: List of stop sequences
        add_think_tags: If True, wraps response with <think> tags before <answer>
        check_reasoning: If True, extracts reasoning field and formats as <think>reasoning</think> <answer>content</answer>
        max_retries: Maximum number of retry attempts

    Returns:
        The generated text response

    Raises:
        Exception: If all retry attempts fail
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

    # Retry logic with exponential backoff
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            break  # Success, exit retry loop
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt  # Exponential backoff: 1s, 2s, 4s
                print(f"\nAPI call failed (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # All retries exhausted
                raise Exception(f"API call failed after {max_retries} attempts: {e}")
    message = result["choices"][0]["message"]
    content = message["content"]

    # Handle reasoning field if check_reasoning is enabled
    if check_reasoning and "reasoning" in message and message["reasoning"]:
        reasoning = message["reasoning"]

        # Wrap reasoning in <think> tags
        think_section = f"<think>{reasoning}</think>"

        # Check if content is already wrapped in <answer> tags
        if "<answer>" not in content:
            content = f"<answer>{content}"
        if "</answer>" not in content:
            content = f"{content}</answer>"

        # Combine think and answer sections
        return f"{think_section} {content}".strip()

    # Add think tags if requested (original behavior)
    if add_think_tags and "<answer>" in content:
        # Add <think> at the beginning and </think> before <answer>
        if "<think>" not in content:
            content = f"<think>{content}"
        if "</think>" not in content:
            content = content.replace("<answer>", "</think> <answer>", 1)
        if "</answer>" not in content:
            content = f"{content}</answer>"

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
    data_idx_start: int = 0,
    data_idx_end: int | None = None,
    add_think_tags: bool = False,
    check_reasoning: bool = False,
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
        check_reasoning: If True, extracts reasoning field and formats as <think>reasoning</think> <answer>content</answer>

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
    if data_idx_start:
        questions, answers = questions[data_idx_start:], answers[data_idx_start:]
    if data_idx_end:
        questions, answers = questions[:data_idx_end], answers[:data_idx_end]
    # Interpolate each question into the system prompt
    prompts = [system_prompt.replace("{question}", question) for question in questions]

    # Collect intermediate results
    intermediate_results = []

    for i, (prompt, answer) in enumerate(zip(prompts, answers)):
        actual_index = data_idx_start + i  # Track actual index in original dataset
        print(f"Processing {i + 1}/{len(prompts)} (index {actual_index})...", end="\r")

        try:
            # Call OpenRouter API with retry logic
            output_text = call_openrouter_api(
                prompt=prompt,
                api_key=api_key,
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                add_think_tags=add_think_tags,
                check_reasoning=check_reasoning,
            )

            # Calculate reward
            score = reward_fn(output_text, answer)
            result = {
                "prompt": prompt,
                "answer": answer,
                "output": output_text,
                "format_reward": score["format_reward"],
                "answer_reward": score["answer_reward"],
                "reward": score["reward"],
            }
            intermediate_results.append(result)

            # Save results incrementally after each successful call
            if output_path:
                save_evaluation_results(intermediate_results, output_path)

        except Exception as e:
            print(f"\n\nFATAL ERROR at index {actual_index}: {e}")
            print(
                f"Failed to process example at index {actual_index} after all retries."
            )
            print(f"To resume, restart with data_idx_start={actual_index}")

            # Save what we have so far
            if output_path and intermediate_results:
                save_evaluation_results(intermediate_results, output_path)
                print(f"Saved {len(intermediate_results)} results before failure.")

            # Exit the program
            raise SystemExit(1)

    print()  # New line after progress indicator

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
            path="./data/MATH/data/train-00000-of-00001.parquet"
        ),
        system_prompt=system_prompt,
        temperature=1.0,
        top_p=1.0,
        max_tokens=2048,
        stop=["</answer>"],
        output_path="./evaluation_results_openrouter.jsonl",
        data_idx_start=0,
        data_idx_end=10,
        add_think_tags=False,  # Disable for check_reasoning
        check_reasoning=True,  # Enable reasoning extraction
    )

    print_example_results(results_openrouter, 5)
