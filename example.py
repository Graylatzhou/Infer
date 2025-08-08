from infer.engine.llm_engine import LLMEngine, SamplingParams
from transformers import AutoTokenizer
import os

def main():
	model_path = "/2023022031/Infer/Qwen"
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	llm = LLMEngine(
		model=model_path
	)
	samplingParams = SamplingParams(
		temperature=0.6,
		max_tokens=8192,
	)
	prompts = [
        "介绍一下你自己",
    ]
	prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
	outputs = llm.generate(prompts, sampling_params=samplingParams)
	for prompt, output in zip(prompts, outputs):
		print("\n")
		print(f"Prompt: {prompt}")
		print(f"Completion: {output['text']!r}")

if __name__ == "__main__":
	main()