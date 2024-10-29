import os
import json


with open("oneround.jsonl", "r") as file_qa:
    data_qa = [json.loads(line) for line in file_qa]

with open("mc.jsonl", "r") as file_mc:
    data_mc = [json.loads(line) for line in file_mc]

for qa in data_qa:
    print(qa.keys())
    text = f"{qa[]}"
    break

