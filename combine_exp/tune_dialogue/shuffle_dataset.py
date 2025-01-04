import os
import json
import random

diagA = "../../exp1/tune_dialogue/data/clean_dialogue_llama.jsonl"
diagB = "../../exp2/tune_dialogue/data/clean_dialogue_case.jsonl"

folder_path = "./data"
output_file = f"{folder_path}/combine_dialogue.jsonl"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def write_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

data_a = read_file(diagA)
data_b = read_file(diagB)

combined_data = data_a + data_b
random.shuffle(combined_data)

write_file(combined_data, output_file)
