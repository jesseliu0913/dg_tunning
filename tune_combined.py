import torch
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


dataset = load_dataset('json', data_files='text_full.jsonl')['train']

train_val_test_split = dataset.train_test_split(test_size=0.2, seed=42) 
val_test_split = train_val_test_split['test'].train_test_split(test_size=0.5, seed=42) 

trainset = train_val_test_split['train']
valset = val_test_split['train']
testset = val_test_split['test']


tokenizer = AutoTokenizer.from_pretrained("epfl-llm/meditron-7b")
model = AutoModelForCausalLM.from_pretrained("epfl-llm/meditron-7b", device_map="auto")


# peft_config = LoraConfig(
#     task_type=TaskType.CAUSAL_LM,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1
# )
# model = get_peft_model(model, peft_config)


class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=256):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['text']

        input_ids = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokenized_input = {key: val.squeeze(0) for key, val in input_ids.items()}

        return tokenized_input


train_dataset = CustomQADataset(trainset, tokenizer)
val_dataset = CustomQADataset(valset, tokenizer)
test_dataset = CustomQADataset(testset, tokenizer)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    return_tensors='pt',
)
    
'''
# Test dataloader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,  
    shuffle=True,
    collate_fn=data_collator  
)
for batch in train_dataloader:
    input_ids = batch['input_ids']
    labels = batch['labels']
    
    print(f"Batch input_ids shape: {input_ids.shape}")
    print(f"Batch labels shape: {labels.shape}")
    
    print(f"Example input_ids: {input_ids[0]}")
    print(f"Example labels: {labels[0]}")
    break
'''

training_args = TrainingArguments(
    output_dir="./meditron_qa_results",
    num_train_epochs=3,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=0.4,
    logging_steps=100,
    learning_rate=2e-5,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-5,
    fp16=False, 
    bf16=True, 
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
trainer.save_model("full_meditron_7b")
