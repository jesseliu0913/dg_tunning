import json
import torch
from torch.utils.data import IterableDataset
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType


class LazyConversationDataset(IterableDataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def __iter__(self):
        return self.example_generator()

    def example_generator(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'clean_dialogue' in data:
                    conversation = data['clean_dialogue']
                    input_texts, output_texts = self.preprocess_conversation(conversation)
                    for input_text, output_text in zip(input_texts, output_texts):
                        input_tokens = self.tokenizer(
                            input_text,
                            truncation=True,
                            max_length=512,
                        )
                        output_tokens = self.tokenizer(
                            output_text,
                            truncation=True,
                            max_length=128,
                        )
                        yield {
                            'input_ids': input_tokens['input_ids'],
                            'attention_mask': input_tokens['attention_mask'],
                            'labels': output_tokens['input_ids'],
                        }

    def preprocess_conversation(self, conversation):
        input_texts = []
        output_texts = []
        history = ""
        for i in range(len(conversation)):
            line = conversation[i].strip()
            if line:
                history += line + " "
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


peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
model = get_peft_model(model, peft_config)

file_path = './clean_dialogue_llama.jsonl'  
train_dataset = LazyConversationDataset(file_path, tokenizer)

from transformers import DataCollatorWithPadding
import torch

class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    def __call__(self, features):
        input_features = [{k: v for k, v in f.items() if k != 'labels'} for f in features]
        label_features = [f['labels'] for f in features]
        
        batch = super().__call__(input_features)
        
        labels = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(l) for l in label_features], 
            batch_first=True, 
            padding_value=self.tokenizer.pad_token_id
        )
        batch['labels'] = labels
        return batch
data_collator = CustomDataCollatorWithPadding(tokenizer=tokenizer)

data_loader = DataLoader(
    train_dataset,
    batch_size=4, 
    collate_fn=data_collator 
)


training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,  # Adjust based on your GPU memory
    gradient_accumulation_steps=1,
    evaluation_strategy='no',  # Set to 'steps' or 'epoch' if you have an evaluation dataset
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    learning_rate=5e-5,
    fp16=torch.cuda.is_available(),  # Enable mixed precision if using a GPU with FP16 support
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()

