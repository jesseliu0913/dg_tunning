import json
import torch
import random
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from peft import PeftModel, LoraConfig, TaskType, get_peft_model
from trl import PPOTrainer, PPOConfig, set_seed
import torch.nn.functional as F


dist.init_process_group(backend='nccl', init_method='env://')
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())

local_rank = dist.get_rank()
world_size = dist.get_world_size()
device = torch.device("cuda", local_rank)

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

# Dataset preparation
class PPOConversationDataset:
    def __init__(self, file_path, tokenizer, max_length=1024):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.examples = []
        self.max_length = max_length
        self._prepare_data()

    def _prepare_data(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if 'clean_dialogue' in data:
                conversation = data['clean_dialogue']
                input_texts, output_texts = self.preprocess_conversation(conversation)
                for input_text, output_text in zip(input_texts, output_texts):
                    self.examples.append((input_text, output_text))

    def preprocess_conversation(self, conversation):
        input_texts, output_texts = [], []
        for i in range(len(conversation)):
            line = conversation[i].strip()
            if line.startswith("Doctor:"):
                if i == 0 or conversation[i - 1].strip().startswith("Doctor:"):
                    continue
                input_text = ' '.join(conversation[:i]).strip()
                output_text = line[len("Doctor:"):].strip()
                input_texts.append(input_text)
                output_texts.append(output_text)
        return input_texts, output_texts

    def __getitem__(self, idx):
        input_text, output_text = self.examples[idx]
        return input_text, output_text

    def __len__(self):
        return len(self.examples)


file_path = './data/clean_dialogue_llama.jsonl'
dataset = PPOConversationDataset(file_path, tokenizer)


ppo_config = PPOConfig(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    learning_rate=5e-6,
    batch_size=8,
    log_with='wandb',
    mini_batch_size=4,
    optimize_clip_ration=0.2,
    total_episodes=len(dataset),
)


def reward_fn_with_logits(output_logits, generated_logits):
    """
    Calculate JSD between output logits and generated logits.
    """
    output_prob = F.softmax(output_logits, dim=-1)
    generated_prob = F.softmax(generated_logits, dim=-1)
    
    mean_prob = 0.5 * (output_prob + generated_prob)
    
    jsd = 0.5 * (F.kl_div(mean_prob.log(), output_prob, reduction='batchmean') +
                 F.kl_div(mean_prob.log(), generated_prob, reduction='batchmean'))
    return -jsd.item()  


ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
)


set_seed(42)
for epoch in range(3):  
    # If using a DistributedSampler, you would call: sampler.set_epoch(epoch)
    for idx, (input_text, output_text) in enumerate(dataset):
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)
        labels = tokenizer(output_text, return_tensors="pt", max_length=512, truncation=True, padding="max_length").to(device)

        with torch.no_grad():
            output_logits = model(**labels).logits

        response = ppo_trainer.generate(inputs['input_ids'], max_new_tokens=50)
        generated_logits = model(input_ids=response).logits

        reward = reward_fn_with_logits(output_logits, generated_logits)

        ppo_trainer.step(inputs['input_ids'], response, torch.tensor([reward]).to(device))

    if local_rank == 0:
        print(f"Epoch {epoch + 1} completed.")

if local_rank == 0:
    with FSDP.state_dict_type(base_model, StateDictType.FULL_STATE_DICT, FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
        full_state_dict = model.state_dict()
        
    torch.save(full_state_dict, "ppo_dialogue_llama_umls.pt")
