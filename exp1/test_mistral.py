import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer

model_weight = "mistralai/Ministral-8B-Instruct-2410"
tokenizer = AutoTokenizer.from_pretrained(model_weight)
model = AutoModelForCausalLM.from_pretrained(model_weight, torch_dtype=torch.float16, device_map="cuda", low_cpu_mem_usage=True, trust_remote_code=True)
print(model)
model.eval()
input_text = f"""
Here is the background information A 23-year-old pregnant woman at 22 weeks gestation presents with burning upon urination.
Question: Which of the following is the best treatment for this patient?
Answer: Nitrofurantoin
Below are several evidence sentences. 
Identify the 4 sentences that, if added to the background information, would support inferring the answer based on the given question-answer pair.
0. She states it started 1 day ago and has been worsening despite drinking more water and taking cranberry extract.
1. She has no complaints, but notes that the new shoes she bought 2 weeks ago do not fit anymore.
2. She otherwise feels well and is followed by a doctor for her pregnancy.
3. Her temperature is 97.7°F (36.5°C), blood pressure is 122/77 mmHg, pulse is 80/min, respirations are 19/min, and oxygen saturation is 98% on room air.
4. Past medical history is significant for a myocardial infarction 6 months ago and NYHA class II chronic heart failure.
5. Her medical history is unremarkable.
6. She has a 15-pound weight gain since the last visit 3 weeks ago.
7. Physical exam is notable for an absence of costovertebral angle tenderness and a gravid uterus.
8. The blood pressure on repeat assessment 4 hours later is 151/90 mm Hg.
Provide only the indices of the relevant sentences in brackets and formatted like this: [ ].
ANSWER: 
"""
inputs = tokenizer(input_text, return_tensors="pt").to('cuda')
output = model.generate(**inputs, max_new_tokens=50, temperature=0.7, top_p=0.9)
print(tokenizer.decode(output[0], skip_special_tokens=True))


