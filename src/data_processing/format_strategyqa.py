import json
from copy import deepcopy
import re


# res_file = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_doubao_response/strategyqa.json", 'r'))
# ori_file = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/train.json", 'r'))
ori_file = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/dev.json", 'r'))

new_res = []
for i in ori_file:
    res = {
        "question": i['question'],
        "answer": i['answer'],
        "facts": i['facts'],
    }
    new_res.append(res)

# json.dump(new_res, open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/train_formatted.json", 'w'), indent=2)
json.dump(new_res, open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/dev_formatted.json", 'w'), indent=2)
