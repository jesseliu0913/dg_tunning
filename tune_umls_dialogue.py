import json
import torch
import random
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import PeftModel, LoraConfig, TaskType, PeftConfig, get_peft_model


class ConversationDataset(Dataset):
    def __init__(self, file_path, tokenizer, split="train", val_split_ratio=0.1, seed=42):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.split = split
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.examples = []
        self._prepare_data()

    def _prepare_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.seed(self.seed)
        random.shuffle(lines)
        split_idx = int(len(lines) * self.val_split_ratio)
        if self.split == "train":
            lines = lines[split_idx:]
        elif self.split == "val":
            lines = lines[:split_idx]
        else:
            raise ValueError("Invalid split value. Use 'train' or 'val'.")

        for line in lines:
            data = json.loads(line)
            if 'clean_dialogue' in data:
                conversation = data['clean_dialogue']
                input_texts, output_texts = self.preprocess_conversation(conversation)
                for input_text, output_text in zip(input_texts, output_texts):
                    full_text = input_text + " " + output_text
                    inputs = self.tokenizer(
                        full_text,
                        truncation=True,
                        max_length=512,
                    )
                    input_tokens = self.tokenizer(
                        input_text,
                        truncation=True,
                        max_length=512,
                    )
                    input_length = len(input_tokens['input_ids'])
                    labels = inputs['input_ids'].copy()
                    labels[:input_length] = [-100] * input_length  # Mask input tokens
                    self.examples.append({
                        'input_ids': inputs['input_ids'],
                        'attention_mask': inputs['attention_mask'],
                        'labels': labels,
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

    def preprocess_conversation(self, conversation):
        input_texts = []
        output_texts = []
        for i in range(len(conversation)):
            line = conversation[i].strip()
            if line:
                if line.startswith("Doctor:"):
                    if i == 0 or conversation[i - 1].strip().startswith("Doctor:"):
                        continue
                    else:
                        input_text = ' '.join(conversation[:i]).strip()
                        output_text = line[len("Doctor:"):].strip()
                        input_texts.append(input_text)
                        output_texts.append(output_text)
        return input_texts, output_texts



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

file_path = './data/clean_dialogue_llama.jsonl'  
train_dataset = ConversationDataset(file_path, tokenizer, split="train")
val_dataset = ConversationDataset(file_path, tokenizer, split="val")



peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

# lora_weights_path = "JesseLiu/umls_dialogue"
# model = PeftModel.from_pretrained(model, lora_weights_path)
# model.train()


class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [feature['labels'] for feature in features]
        features = [{k: v for k, v in feature.items() if k != 'labels'} for feature in features]
        batch = super().__call__(features)
        max_label_length = max(len(l) for l in labels)
        padded_labels = torch.full((len(labels), max_label_length), -100)
        for i, label in enumerate(labels):
            padded_labels[i, :len(label)] = torch.tensor(label)
        batch['labels'] = padded_labels
        return batch

data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

# data_loader = DataLoader(
#     train_dataset,
#     batch_size=4, 
#     collate_fn=data_collator 
# )


fsdp_config = {
    "fsdp_min_num_params": 20000,
    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    "fsdp_sharding_strategy": "FULL_SHARD",
}

training_args = TrainingArguments(
    output_dir="./llama_dialogue_results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=2e-5,
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
    save_total_limit=5,
    report_to='wandb',
    ddp_find_unused_parameters=False,
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
trainer.save_model("dialogue_llama_umls")

