import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class CustomLLM(torch.nn.Module):
    def __init__(self, samples_per_prompt: int, 
                 device, 
                 model_name="bigcode/starcoder2-15b-instruct-v0.1", 
                 quantization_config=None, 
                 log_path=None):
        super().__init__()
        self._samples_per_prompt = samples_per_prompt
        self.prompt_count = 0
        self.log_path = log_path
        self.device = device
        self.device_map = {"": self.device}
        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              trust_remote_code=True, 
                                                              quantization_config=quantization_config,
                                                              low_cpu_mem_usage=True,
                                                              device_map=self.device_map)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                              trust_remote_code=True,
                                                              low_cpu_mem_usage=True,
                                                              device_map=self.device_map)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = "[PAD]" 
        self.tokenizer.pad_token_id = self.tokenizer.convert_tokens_to_ids("[PAD]")
        # self.tokenizer.pad_token = self.tokenizer.eos_token
        self.system_prompt = ("You are an exceptionally intelligent coding assistant" 
                            "that consistently delivers accurate and reliable responses to user instructions.\n"
                            "### Instruction\n\n")
        self.response_prompt = "\n### Response\n"
        
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def draw_samples(self, prompt: str, max_length=800):
        # print("Model is on device:", self.model.device)
        prompt = self.system_prompt + prompt + self.response_prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt', padding=True)
        input_ids = input_ids.to(self.model.device)
        
        samples = []
        for _ in range(self._samples_per_prompt):
            output = self.model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=1, 
                no_repeat_ngram_size=None, 
                do_sample=True, 
                top_k=40, 
                top_p=0.95, 
                repetition_penalty=1.1,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()

            samples.append(response)
            self._log(prompt, response, self.prompt_count)
            self.prompt_count += 1
        
        return samples

    def _log(self, prompt: str, response: str, index: int):
        if self.log_path is not None:
            with open(self.log_path / f"prompt_{index}.log", "a") as f: # saves the prompt in file
                f.write(prompt)
            with open(self.log_path / f"response_{index}.log", "a") as f:
                f.write(str(response))

# Example usage
if __name__ == "__main__":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    llm = CustomLLM(samples_per_prompt=1, device="cuda:1", quantization_config=quantization_config)
    # prompt = 
    
    # prompt = ("### Instruction\n def sum_first_n(n):\n"
    #           "# fill here\n"
    #           "return out\n"
    #           "### Response")
    prompt = "def fibonacci(n):\n"
             
    samples = llm.draw_samples(prompt=prompt)
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}: {sample}")