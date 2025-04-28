import os
import json
import torch
import gdown
import random
import argparse
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset
from peft import PeftModel, LoraConfig, TaskType, PeftConfig, get_peft_model


DATA_FOLDER = "./data"
if not os.path.exists(DATA_FOLDER):
    os.makedirs(DATA_FOLDER)
    file_id = "1FIVsfphH1O7tFwa-HnkPxOnqHoTzshDR"
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, f"{DATA_FOLDER}/raw_case.jsonl", quiet=False)
    
contextset = load_dataset('json', data_files=f'{DATA_FOLDER}/raw_case.jsonl')['train']
context_split = contextset.train_test_split(test_size=0.1, seed=42) 
train_context = context_split['train']
val_context = context_split['test']

mcset = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
mc_split = mcset.train_test_split(test_size=0.1, seed=42) 
train_mc = mc_split['train'] 
val_mc = mc_split['test']

parser = argparse.ArgumentParser(description="Casual Tuning based on case report")
parser.add_argument("--model", type=str, default=None, help="Set model weights")
parser.add_argument("--epoch", type=int, default=3, help="Set Epoch")
parser.add_argument("--task", type=str, default="llama3.2", help="Set Task Name")
parser.add_argument("--batch_size", type=int, default=4, help="Set Batch Size")
parser.add_argument("--max_length", type=int, default=2048, help="Set max_length")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Set Learning Rate")

args = parser.parse_args()

class CombinedQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048, mode='context'):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        if self.mode == 'multi_choice':
            # Multi-choice format
            question = item['question']
            options = item['options']
            correct_answer = item['answer_idx']

            input_text = f"Question: {question}\nOptions:\n"
            for idx, option in enumerate(options):
                input_text += f"{idx}: {option}\n"
            input_text += "Answer: "

            target_text = f"{correct_answer}"
            full_text = input_text + target_text

            tokenized = self.tokenizer(
                full_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = tokenized.attention_mask.squeeze(0)

            labels = input_ids.clone()
            answer_start_idx = len(self.tokenizer.encode(input_text, add_special_tokens=False))
            labels[:answer_start_idx] = -100  

        elif self.mode == 'context':
            # Report tuning format
            input_text = str(item['case'])

            tokenized = self.tokenizer(
                input_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized.input_ids.squeeze(0)
            attention_mask = tokenized.attention_mask.squeeze(0)

            labels = input_ids.clone() 

        else:
            raise ValueError("Unsupport value")

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
    padding='longest',
    return_tensors='pt'
)

mc_train_dataset = CombinedQADataset(
    data=train_mc,
    tokenizer=tokenizer,
    max_length=args.max_length,
    mode='multi_choice'
)
mc_val_dataset = CombinedQADataset(
    data=val_mc,
    tokenizer=tokenizer,
    max_length=args.max_length,
    mode='multi_choice'
)
context_train_dataset = CombinedQADataset(
    data=train_context,
    tokenizer=tokenizer,
    max_length=args.max_length,
    mode='context'
)
context_val_dataset = CombinedQADataset(
    data=val_context,
    tokenizer=tokenizer,
    max_length=args.max_length,
    mode='context'
)

train_dataset = ConcatDataset([mc_train_dataset, context_train_dataset])
val_dataset = ConcatDataset([mc_val_dataset, context_val_dataset])


'''
# Test dataloader
train_dataloader = DataLoader(
    mc_train_dataset,
    batch_size=1,
    shuffle=True,
    collate_fn=data_collator
)

val_dataloader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=data_collator
)

for idx, item in enumerate(train_dataloader):
    print(idx, item)
    break
'''

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

folder_path = "./model_weights"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

training_args = TrainingArguments(
    output_dir=f"{folder_path}/{args.task}_inter",
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=2,
    # gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=0.4,
    logging_steps=10,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    weight_decay=0.01,
    max_grad_norm=0.5,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-5,
    ddp_backend='nccl',
    fp16=False, 
    bf16=False, 
    # fsdp='full_shard auto_wrap',
    # fsdp_config=fsdp_config,
    # deepspeed="ds_config.json",
    save_total_limit=5,
    report_to='wandb',
    ddp_find_unused_parameters=False  
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
trainer.save_model(f"{folder_path}/{args.task}_final")
