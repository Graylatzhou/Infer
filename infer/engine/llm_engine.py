import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from infer.config import Config
from infer.engine.sequence import Sequence, SamplingParams
from infer.engine.scheduler import Scheduler
from infer.engine.model_runner import ModelRunner
from infer.ops.sampler import Sampler

class LLMEngine:
	def __init__(self, model, **kwargs):
		config_fields = {field.name for field in fields(Config)}
		config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
		config = Config(model, **config_kwargs)
		self.model_runner = ModelRunner(config)
		self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
		config.eos = self.tokenizer.eos_token_id
		self.scheduler = Scheduler(config)
		atexit.register(self._cleanup)

	def _cleanup(self):
		del self.model_runner.graphs, self.model_runner.graph_pool
		del self.model_runner

	def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
		if isinstance(prompt, str):
			prompt = self.tokenizer.encode(prompt)
		seq = Sequence(prompt, sampling_params)
		self.scheduler.add(seq)

	def is_finished(self):
		return self.scheduler.is_finished()
	
	def step(self):
		seqs, is_prefill = self.scheduler.schedule()
		token_ids = self.model_runner.run(seqs, is_prefill)
		self.scheduler.postprocess(seqs, token_ids)
		outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
		# minus token num means decode
		num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
		return outputs, num_tokens
	
	def generate(
			self, 
			prompts: list[str] | list[list[int]], 
			sampling_params: SamplingParams | list[SamplingParams],
			use_tqdm: bool = True,
		):
		if use_tqdm:
			pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
		if not isinstance(sampling_params, list):
			sampling_params = [sampling_params] * len(prompts)
		for prompt, sampling_param in zip(prompts, sampling_params):
			self.add_request(prompt, sampling_param)
		output_dict = dict()
		prefill_throughput = decode_throughput = 0
		while not self.is_finished():
			t = perf_counter()
			output, num_tokens = self.step()
			if use_tqdm:
				if num_tokens > 0:
					prefill_throughput = num_tokens / (perf_counter() - t)
				else:
					decode_throughput = -num_tokens / (perf_counter() - t)
				pbar.set_postfix(
					{
						"prefill_throughput": f"{prefill_throughput:.2f} tokens/s",
						"decode_throughput": f"{decode_throughput:.2f} tokens/s",
					}
				)
			for seq_id, token_ids in output:
				output_dict[seq_id] = token_ids
				if use_tqdm:
					pbar.update(1)
		outputs = [output_dict[seq_id] for seq_id in sorted(output_dict.keys())]
		outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
		if use_tqdm:
			pbar.close()
		return outputs

