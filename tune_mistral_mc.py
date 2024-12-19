import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset


trainset = load_dataset('GBaker/MedQA-USMLE-4-options', split='train')
train_val_test_split = trainset.train_test_split(test_size=0.1, seed=42) 
trainset = train_val_test_split['train'] 
valset = train_val_test_split['test']
testset = load_dataset('GBaker/MedQA-USMLE-4-options', split='test')


tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3")


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)
# lora_weights_path = "JesseLiu/umls_mc"
# model = PeftModel.from_pretrained(model, lora_weights_path)


class CustomQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = data
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
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

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


train_dataset = CustomQADataset(trainset, tokenizer)
val_dataset = CustomQADataset(valset, tokenizer)
test_dataset = CustomQADataset(testset, tokenizer)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest', return_tensors='pt')

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(
#     train_dataset,
#     batch_size=2,  
#     shuffle=True,
#     collate_fn=data_collator  
# )

# batch = next(iter(train_dataloader))
# print(f"Input IDs shape: {batch['input_ids'].shape}")
# print(f"Labels shape: {batch['labels'].shape}")
# print(f"Attention mask shape: {batch['attention_mask'].shape}")
# print("First example input IDs:", batch['input_ids'][0])
# print("First example labels:", batch['labels'][0])

fsdp_config = {
    "fsdp_min_num_params": 20000,
    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    "fsdp_sharding_strategy": "FULL_SHARD",
}

training_args = TrainingArguments(
    output_dir="./mistral_mc_results",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    # save_steps=0.4,
    logging_steps=100,
    learning_rate=5e-4,
    warmup_ratio=0.1,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="cosine",
    adam_beta1=0.9,
    adam_beta2=0.95,
    adam_epsilon=1e-5,
    ddp_backend='nccl',
    fp16=False, 
    bf16=True, 
    fsdp='full_shard auto_wrap',
    fsdp_config=fsdp_config,
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
trainer.save_model("mistral7b_mc")

# CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node=4 tune_mistral_mc.py > ./log/mc.log 2>&1 &
