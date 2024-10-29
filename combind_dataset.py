import os
import json
import re

def remove_special_symbols(text):
    cleaned_text = cleaned_text.replace("*", "").replace("-", "").replace("_", "").replace("**", "")
    return cleaned_text

with open("oneround.jsonl", "r") as file_qa:
    data_qa = [json.loads(line) for line in file_qa]

with open("mc.jsonl", "r") as file_mc:
    data_mc = [json.loads(line) for line in file_mc]

for qa in data_qa:
    qa_dict = {}
    context = qa['context']
    question = qa['question']
    answer = qa['answer']
    input_text = f"{context}\n {question}\n Answer: {answer} "
    qa_dict['text'] = remove_special_symbols(input_text)
    
    with open("text_full.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(qa_dict) + "\n")

for mc in data_mc:
    mc_dict = {}
    context = mc['context']
    question = mc['question']
    answer = mc['answer']
    input_text = f"{context}\n {question}\n Answer: {answer} "
    mc_dict['text'] = remove_special_symbols(input_text)
    # print(mc_dict)
    
    with open("text_full.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(mc_dict) + "\n")


