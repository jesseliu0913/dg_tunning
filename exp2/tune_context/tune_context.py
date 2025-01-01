import os
import torch
import argparse
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling 
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


parser = argparse.ArgumentParser(description="Casual Tuning based on case report")
parser.add_argument("--model", type=str, default=None, help="Set model weights")
parser.add_argument("--epoch", type=int, default=3, help="Set Epoch")
parser.add_argument("--task", type=str, default="llama3.2", help="Set Task Name")
parser.add_argument("--batch_size", type=int, default=4, help="Set Batch Size")
parser.add_argument("--max_length", type=int, default=2048, help="Set max_length")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Set Learning Rate")

args = parser.parse_args()


dataset = load_dataset('json', data_files='./data/raw_case.jsonl')['train']
train_val_test_split = dataset.train_test_split(test_size=0.1, seed=42) 
trainset = train_val_test_split['train']
valset = train_val_test_split['test']

tokenizer = AutoTokenizer.from_pretrained(args.model)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(args.model)
# model.gradient_checkpointing_enable()  

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = str(item['case'])

        input_ids = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokenized_input = {key: val.squeeze(0) for key, val in input_ids.items()}

        return tokenized_input


train_dataset = CustomQADataset(trainset, tokenizer, max_length=args.max_length)
val_dataset = CustomQADataset(valset, tokenizer, max_length=args.max_length)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors='pt',
)
    
"""
# Test dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,  
    shuffle=True,
    collate_fn=data_collator  
)
for batch in train_dataloader:
    print(batch)
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    print(f"Example input_ids: {input_ids[0]}")
    print(f"Example labels: {labels[0]}")
    break
"""

# fsdp_config = {
#     "fsdp_min_num_params": 20000,
#     "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
#     "fsdp_sharding_strategy": "FULL_SHARD",
# }
folder_path = "./model_weights"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

training_args = TrainingArguments(
    output_dir=f"{folder_path}/{args.task}_inter",
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    gradient_accumulation_steps=6,
    # gradient_checkpointing=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=0.4,
    logging_steps=10,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
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
