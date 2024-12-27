import os
import json


file1 = './data/raw_case1.jsonl'
file2 = './data/raw_case2.jsonl'
output_file = 'raw_case.jsonl'

combine_data = []

with open(file1, "r") as f1_read:
  for line in f1_read:
    combine_data.append(json.loads(line))

with open(file2, "r") as f2_read:
  for line in f2_read:
    combine_data.append(json.loads(line))

with open(output_file, "w") as f_out:
  for item in combine_data:
    f_out.write(json.dumps(item) + "\n")