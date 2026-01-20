from typing import Callable, List
from vllm import LLM, SamplingParams
from drgrpo_grader import r1_zero_reward_fn


def evaluate_llama(
    model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    eval_sampling_params: SamplingParams,
):
    pass


qwen = LLM(model="Qwen/Qwen2.5-Math-1.5B")
eval_sampling_params = SamplingParams(
    temperature=1.0, top_p=1.0, max_tokens=1024, stop=["\n"]
)
