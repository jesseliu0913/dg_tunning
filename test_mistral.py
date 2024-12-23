import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

model_weight = "rellabear/dialogue_mistral_umls"
tokenizer = AutoTokenizer.from_pretrained(model_weight)
model = AutoModelForCausalLM.from_pretrained(model_weight, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True, trust_remote_code=True)

model.eval()
input_text = "Patient: I have a headache. Doctor:"
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
output = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))


