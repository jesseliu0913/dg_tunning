import os
import json
import re

def remove_special_symbols(text):
    cleaned_text = text.replace("*", "").replace("-", "").replace("_", "").replace("**", "")
    return cleaned_text

with open("oneround.jsonl", "r") as file_qa:
    data_qa = [json.loads(line) for line in file_qa]

with open("mc.jsonl", "r") as file_mc:
    data_mc = [json.loads(line) for line in file_mc]


mc_prefix = """
<|im_start|>system
You are a highly knowledgeable medical specialist tasked with answering multiple-choice questions based on rare medical case reports. Utilize your extensive understanding of rare medical conditions, clinical presentations, diagnostic imaging findings, laboratory results, and current medical guidelines to determine the most accurate diagnosis. When answering, consider the patient's history, clinical findings, and any imaging or diagnostic results presented in each case. Select one correct answer from the given options based on the standard practices and the latest advancements in medical knowledge. 
<|im_end|>
"""

qa_prefix = """
<|im_start|>system
You are a medical expert specializing in rare diseases and complex medical cases. Your task is to analyze clinical case reports and provide accurate answers based on the patient's presentation, medical history, and clinical findings. Each case is unique and may involve uncommon conditions requiring specialized knowledge. Your responses should be concise and well-supported by medical guidelines or established clinical practices. Base your answers on your understanding of the mechanisms of disease, clinical science, diagnostic criteria, and treatment protocols. Carefully consider the context provided in each case before arriving at a diagnosis.
<|im_end|>
"""

for qa in data_qa:
    qa_dict = {}
    context = remove_special_symbols(qa['context'])
    question = "What is the most likely diagnosis?"
    answer = qa['answer']
    input_text = f"""
<|im_start|>question
{context}
Question: {question}<|im_end|>
<|im_start|>answer
{answer} 
<|im_end|>
"""
    
    qa_dict['text'] = qa_prefix + input_text
    
    with open("text_full.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(qa_dict) + "\n")

    with open("oneround_dealed.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(qa_dict) + "\n")

for mc in data_mc:
    mc_dict = {}
    context = remove_special_symbols(mc['context'])
    question = remove_special_symbols(mc['question'])
    answer = mc['answer']
    input_text = f"""
<|im_start|>question
{context}
Question: {question}
<|im_end|>
<|im_start|>answer
{answer} 
<|im_end|>
"""
    
    mc_dict['text'] = mc_prefix + input_text
    # print(mc_dict)
    
    with open("text_full.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(mc_dict) + "\n")

    with open("mc_dealed.jsonl", "a+") as jsonl_file:
        jsonl_file.write(json.dumps(mc_dict) + "\n")


