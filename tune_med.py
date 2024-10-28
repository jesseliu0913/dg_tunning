import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling 
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


dataset = load_dataset('json', data_files='oneround.jsonl')['train']
train_val_test_split = dataset.train_test_split(test_size=0.2, seed=42) 
val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42) 

train_dataset_raw = train_val_test_split['train']
val_dataset_raw = val_test_split['train']
test_dataset_raw = val_test_split['test']


tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", device_map="auto")


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)


class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']

        input_text = f"{context}\n {question}\n Answer: "

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            truncation=True,
        )


        labels = self.tokenizer(
            answer,
            max_length=self.max_length,
            truncation=True,
        )

        return {
            'input_ids': inputs.input_ids,
            'labels': labels.input_ids
        }

train_dataset = CustomQADataset(train_dataset_raw, tokenizer)
val_dataset = CustomQADataset(val_dataset_raw, tokenizer)
test_dataset = CustomQADataset(test_dataset_raw, tokenizer)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors='pt',
)


training_args = TrainingArguments(
    output_dir="./meditron_qa_results",
    num_train_epochs=100,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    gradient_accumulation_steps=2,
    evaluation_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    fp16=True,
    save_total_limit=2,
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


trainer.train()
trainer.save_model("oneround_meditron_7b")
