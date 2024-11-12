import torch
import logging
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP
# import os
from transformers import BitsAndBytesConfig
import numpy as np
from custom_llm import CustomLLM  # Import our StarCoder2

from collections.abc import Collection, Sequence
from funsearch import evaluator
from funsearch import programs_database

class CustomSampler:
    """Node that samples program continuations and sends them for analysis."""

    def __init__(
        self,
        # num_gpus=2
        rank: int,
        # database: programs_database.ProgramsDatabase,
        # evaluators: Sequence[evaluator.Evaluator],
        database = None,
        evaluators = None,
        samples_per_prompt = 1,
        quantization_config = None
    ) -> None:
        self._database = database
        self._evaluators = evaluators
        self._samples_per_prompt = samples_per_prompt
        self._rank = rank
        self.device = f"cuda:{self._rank}"
        torch.cuda.set_device(rank)

        self._llm = CustomLLM(samples_per_prompt=self._samples_per_prompt,
                              device=self.device,
                              model_name="bigcode/starcoder2-15b-instruct-v0.1",
                              quantization_config=quantization_config)

        # self._llm = DDP(self._llm, device_ids=[rank]) # only needed if we are doing some training

    def cleanup(self):
        """Clean up any resources allocated by the sampler."""
        logging.info(f"Cleaning up GPU {self._rank}.")
        torch.distributed.destroy_process_group()

    def sample(self):

        prompt = self._database.get_prompt()

        if not prompt:
            logging.info(f"No prompt from database for gpu {self._rank}.")
            return
        
        samples = self._llm.draw_samples(prompt.code)

        for sample in samples:
            chosen_evaluator = np.random.choice(self._evaluators)
            chosen_evaluator.analyse(
                sample, prompt.island_id, prompt.version_generated)
    
    def sample_test(self, prompt="def fibonacci(n):"):

        if not prompt:
            logging.info(f"No prompt from database for gpu {self._rank}.")
            return
        
        samples = self._llm.draw_samples(prompt)

        return samples


# Usage
if __name__ == "__main__":
    # database = programs_database.ProgramsDatabase()
    # evaluators = [evaluator.Evaluator() for _ in range(4)]
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    sampler = CustomSampler(rank=1, quantization_config=quantization_config)
    samples = sampler.sample_test()
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}: {sample}")