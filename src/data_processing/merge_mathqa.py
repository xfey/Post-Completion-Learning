import json
from copy import deepcopy
import re



input_file = "/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response/output_doubao_output3_bi-20250722224741-nqt8l_output_results.jsonl"
question_file = "/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response/input_doubao_offline_trainval3.jsonl"
raw_file = "/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response/raw_mathqa.json"


with open(raw_file, "r") as f:
    raw_data = json.load(f)

questions = {}
answers = {}
rationale = {}
question_to_ids = {}
with open(question_file, "r") as f:
    for i,line in enumerate(f):
        line_data = json.loads(line)
        questions[line_data["custom_id"]] = line_data["body"]["messages"][1]["content"]  ## user question, with \nOptions already
        assert raw_data[i]["Problem"] == line_data["body"]["messages"][1]["content"].split('\nOption')[0]
        answers[line_data["custom_id"]] = raw_data[i]["correct"]
        reasoning = raw_data[i]["Rationale"]
        if reasoning.startswith('\"'):
            reasoning = reasoning[1:]
        if reasoning.endswith('\"'):
            reasoning = reasoning[:-1]
        rationale[line_data["custom_id"]] = reasoning
        # question_raw = line_data["body"]["messages"][1]["content"].split('\nOption')[0]
        # question_to_ids[question_raw] = line_data["custom_id"]


data = []
with open(input_file, "r") as f:    
    for line in f:
        line_data = json.loads(line)
        content = line_data["response"]["body"]["choices"][0]["message"]["content"]
        content = content.replace('<thinking>', '<think>').replace('</thinking>', '</think>').replace('<answering>', '<answer>').replace('</answering>', '</answer>')
        saved_line_data = {
            # "custom_id": line_data["custom_id"],
            "response": content,
            "question": questions[line_data["custom_id"]],
            "answer": answers[line_data["custom_id"]],
            "rationale": rationale[line_data["custom_id"]],
        }
        data.append(saved_line_data)



# rewrew checking



def accuracy_reward(content, sol, **kwargs) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    gold_parsed = sol
    answer_parsed = parse_gsm8k_answer(
        content,
    )
    # extract <answer> from content
    # We require the answer to be provided in correct latex (no malformed operators)
    # answer_parsed = parse(content, extraction_mode="first_match")
    try:
        # print(f"content: {content}, sol: {sol} answer: {answer_parsed}, gold: {gold_parsed}")
        reward = float(verify_gsm8k(answer_parsed, gold_parsed))
        # reward = float(verify(gold_parsed, answer_parsed))
    except Exception as e:
        print(f"verify (gsm8k) failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
        reward = None
    return reward

def parse_gsm8k_answer(text, mode="first_match"):
    pattern = r"<answer>(.*?)</answer>"
    results = re.findall(pattern, text, re.DOTALL)

    if isinstance(results, list) and len(results) > 0:
        # if mode == "first_match":
        results = results[0]
        # else:
        #     results = results[-1]  
    else:
        results = ""
    results = results.strip()
    return results


def verify_gsm8k(answer_parsed, gold_parsed):
    # if gold_parsed.lower() == 'true':
    #     gold_parsed = True
    # elif gold_parsed.lower() == 'false':
    #     gold_parsed = False
    
    if answer_parsed.strip() == gold_parsed.strip():
        return True
    return False



def format_reward_all(content, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>.*?</think>\n?<answer>.*?</answer>\n?<evaluation>.*?</evaluation>\n?<reward>.*?</reward>"    # note: remove the suffix "$", not ends with answer
    matches = re.match(pattern, content, re.DOTALL | re.MULTILINE)
    return 1 if matches else 0


def rewarding_reward_raw(completion, **kwargs):
    try:
        pattern = r"<reward>.*?</reward>"
        results = re.findall(pattern, completion, re.DOTALL)
        if isinstance(results, list) and len(results) > 0:
            # assert has only one result
            results = results[-1]
            results = results[len("<reward>"):-len("</reward>")].strip()
            results = eval(results)
            return {'dt': results, 'valid': True}
        else:
            return {'valid': False}
    except:
        return {'valid': False}




evaluation_accuracy = 0
evaluation_format = 0
evaluation_rewrew = 0


for item in data:
    # question, answer, response
    item["stat_accuracy_reward"] = accuracy_reward(item["response"], item["answer"])
    item["stat_format_all_reward"] = format_reward_all(item["response"])
    rewrew = rewarding_reward_raw(item["response"])

    if rewrew['valid'] == False:
        item["stat_rewarding_reward"] = []
    else:
        item["stat_rewarding_reward"] = rewrew['dt']

    if item["stat_rewarding_reward"] and len(item["stat_rewarding_reward"]) == 2:
        if item["stat_rewarding_reward"][0] == item["stat_accuracy_reward"] and item["stat_rewarding_reward"][1] == item["stat_format_all_reward"]:
            item["stat_rewrew_right"] = True
        else:
            item["stat_rewrew_right"] = False
    item["stat_rewrew_right"] = item.get("stat_rewrew_right", False)

    evaluation_accuracy += item["stat_accuracy_reward"]
    evaluation_format += item["stat_format_all_reward"]
    evaluation_rewrew += int(item["stat_rewrew_right"])



# print stat results
print(f"accuracy: {evaluation_accuracy} / {len(data)}")
print(f"format: {evaluation_format} / {len(data)}")
print(f"rewrew: {evaluation_rewrew} / {len(data)}")



# save
with open("/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response/mathqa_FIN.jsonl", "w") as f:
    for line_data in data:
        f.write(json.dumps(line_data, ensure_ascii=False) + "\n")
