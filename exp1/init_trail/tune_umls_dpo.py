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
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


from datasets import Dataset
import random
import json

class ConversationDataset:
    def __init__(self, file_path, tokenizer, data_split="train", val_split_ratio=0.1, seed=42):
        self.file_path = file_path
        self.tokenizer = tokenizer  
        self.data_split = data_split
        self.val_split_ratio = val_split_ratio
        self.seed = seed
        self.examples = []
        self._prepare_data()

        # Now create a Hugging Face Dataset from the examples list
        self.dataset = Dataset.from_list(self.examples)

    def _prepare_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        random.seed(self.seed)
        random.shuffle(lines)
        split_idx = int(len(lines) * self.val_split_ratio)

        if self.data_split == "train":
            lines = lines[split_idx:]
        elif self.data_split == "val":
            lines = lines[:split_idx]
        else:
            raise ValueError("Invalid split value. Use 'train' or 'val'.")

        for line in lines:
            data = json.loads(line)
            if 'clean_dialogue' in data:
                conversation = data['clean_dialogue']
                prompts, chosens, rejecteds = self.preprocess_conversation(conversation)
                for p, c, r in zip(prompts, chosens, rejecteds):
                    self.examples.append({
                        "prompt": p,
                        "chosen": c,
                        "rejected": r
                    })

    def preprocess_conversation(self, conversation):
        doctor_lines = [line for line in conversation if line.strip().startswith("Doctor:")]

        prompts = []
        chosens = []
        rejecteds = []

        for i in range(len(conversation)):
            line = conversation[i].strip()
            if line.startswith("Doctor:") and i > 0 and conversation[i-1].strip().startswith("Patient:"):
                chosen_line = line[len("Doctor:"):].strip()
                prompt = "\n".join(conversation[:i]).strip()

                possible_rejects = [d for d in doctor_lines if d != conversation[i]]
                if not possible_rejects:
                    continue
                rejected_line_full = random.choice(possible_rejects)
                rejected_line = rejected_line_full[len("Doctor:"):].strip()

                prompts.append(prompt)
                chosens.append(chosen_line)
                rejecteds.append(rejected_line)

        return prompts, chosens, rejecteds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __getattr__(self, name):
        return getattr(self.dataset, name)



tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

class DPODataCollator:
    def __init__(self, tokenizer, max_length=1024):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        prompts = [x["prompt"] for x in batch]
        chosens = [x["chosen"] for x in batch]
        rejecteds = [x["rejected"] for x in batch]
        
        tokenized_prompts = self.tokenizer(prompts, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        tokenized_chosens = self.tokenizer(chosens, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
        tokenized_rejecteds = self.tokenizer(rejecteds, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt")
    
        return {
            "prompt_input_ids": tokenized_prompts["input_ids"],
            "prompt_attention_mask": tokenized_prompts["attention_mask"],
            "chosen_input_ids": tokenized_chosens["input_ids"],
            "chosen_attention_mask": tokenized_chosens["attention_mask"],
            "rejected_input_ids": tokenized_rejecteds["input_ids"],
            "rejected_attention_mask": tokenized_rejecteds["attention_mask"]
        }


data_collator = DPODataCollator(tokenizer=tokenizer)



# train_dataloader = DataLoader(
#     dataset=train_dataset,
#     batch_size=2,
#     shuffle=True,
#     collate_fn=data_collator
# )


# for batch in train_dataloader:
#     print(batch)
#     break  
    
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

file_path = './data/clean_dialogue_llama.jsonl'  
train_dataset = ConversationDataset(file_path, tokenizer, data_split="train")
val_dataset = ConversationDataset(file_path, tokenizer, data_split="val")
# hf_train_dataset = train_dataset.to_hf_dataset()
# hf_val_dataset = val_dataset.to_hf_dataset()



peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)




fsdp_config = {
    "fsdp_min_num_params": 20000,
    "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
    "fsdp_sharding_strategy": "FULL_SHARD",
}

training_args = DPOConfig(
    output_dir="./llama_dialogue_results",
    num_train_epochs=5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    max_prompt_length=512,
    max_completion_length=512,
    save_strategy="epoch",
    logging_steps=50,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    warmup_steps=500,
    weight_decay=0.1,
    max_grad_norm=1.0,
    lr_scheduler_type="constant_with_warmup",
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


trainer = DPOTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    # data_collator=data_collator,
)


trainer.train()
trainer.save_model("dialogue_llama_umls_dpo")
