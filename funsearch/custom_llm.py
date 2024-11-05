import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class CustomLLM(torch.nn.Module):
    def __init__(self, samples_per_prompt: int, device, model_name="bigcode/starcoder2-15b-instruct-v0.1", quantization_config=None, log_path=None):
        super().__init__()
        self._samples_per_prompt = samples_per_prompt
        self.prompt_count = 0
        self.log_path = log_path
        if quantization_config is not None:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, quantization_config=quantization_config)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(torch.device(device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def draw_samples(self, prompt, max_length=400):
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
        input_ids = input_ids.to(self.model.device)
        
        samples = []
        for _ in range(self._samples_per_prompt):
            output = self.model.generate(
                input_ids, 
                max_length=max_length, 
                num_return_sequences=1, 
                no_repeat_ngram_size=3, 
                do_sample=True, 
                # top_k=50, 
                top_p=0.95, 
                temperature=1
            )
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
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
    llm = CustomLLM(samples_per_prompt=1, device="cuda:0", model_name="bigcode/starcoder2-3b", quantization_config=quantization_config)
    prompt = "def fibonacci(n):"
    samples = llm.draw_samples(prompt=prompt)
    for i, sample in enumerate(samples, 1):
        print(f"Sample {i}: {sample}")