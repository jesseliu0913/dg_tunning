import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


dataset = load_dataset("GBaker/MedQA-USMLE-4-options")
train_val_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_val_split["train"]
val_dataset = train_val_split["test"]
test_dataset = dataset["test"]  


def preprocess(example):
    options = " ".join([f"({key}) {value}" for key, value in example['options'].items()])
    input_text = f"Question: {example['question']} Options: {options} Answer: {example['answer']}"
    return {
        "input_text": input_text
    }

train_dataset = train_dataset.map(preprocess)
val_dataset = val_dataset.map(preprocess)
if test_dataset:
    test_dataset = test_dataset.map(preprocess)


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token  

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)


class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input_text']

        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )


        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()


        answer_start = input_text.find('Answer:')
        prompt_encoding = self.tokenizer(
            input_text[:answer_start + len('Answer:')],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        prompt_length = len(prompt_encoding['input_ids'].squeeze())


        labels = input_ids.clone()
        labels[:prompt_length] = -100  

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


train_dataset = CustomQADataset(train_dataset, tokenizer)
val_dataset = CustomQADataset(val_dataset, tokenizer)
if test_dataset:
    test_dataset = CustomQADataset(test_dataset, tokenizer)


training_args = TrainingArguments(
    output_dir="./llama_qa_results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    fp16=False,
    bf16=True,
    save_total_limit=5,
    report_to='none',  
    ddp_find_unused_parameters=False,
    fsdp='full_shard auto_wrap',
    fsdp_transformer_layer_cls_to_wrap='LlamaDecoderLayer',
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model("MC_llama_3b_instruct")
