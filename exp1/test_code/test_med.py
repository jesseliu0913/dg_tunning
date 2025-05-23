from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from tqdm import tqdm
from datasets import load_dataset


dataset = load_dataset('json', data_files='oneround.jsonl')['train']
train_val_test_split = dataset.train_test_split(test_size=0.2, seed=42) 
val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42) 

trainset = train_val_test_split['train']
valset = val_test_split['train']
testset = val_test_split['test']


tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
base_model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", device_map="auto")
model = PeftModel.from_pretrained(base_model, "oneround_meditron_7b")
model.eval()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


generated_answers = []


for item in tqdm(testset):
    context = item['context']
    question = item['question']
    input_text = f"{context}\n {question}\n Answer: "
    print(input_text)

    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)


    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )
    

    generated_text = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)
    generated_answers.append(generated_text.strip())
    print(generated_answers)
    break


# for i, item in enumerate(testset):
#     print(f"Sample {i+1}:")
#     print(f"Context: {item['context']}")
#     print(f"Question: {item['question']}")
#     print(f"Generated Answer: {generated_answers[i]}")
#     print(f"Reference Answer: {item['answer']}")
#     print('-' * 80)
