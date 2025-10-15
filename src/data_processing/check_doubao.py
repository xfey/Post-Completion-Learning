import json
from datasets import load_dataset
import re

input_file = "/mnt/bn/ocr-rl-data/whiteboxRL/data/gsm8k_doubao_response_FIN.jsonl"

all_data = []
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        all_data.append(data)

for item in all_data:
    if "<evaluation>" not in item['response']:
        print(item['response'])
