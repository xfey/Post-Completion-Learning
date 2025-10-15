import json
from datasets import load_dataset
import re


def accuracy_reward(content, sol, **kwargs) -> list[float]:
    """Reward function that checks if the completion is the same as the ground truth."""
    gold_parsed = sol
    if len(gold_parsed) != 0:
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
    else:
        # If the gold solution is not parseable, we assign `None` to skip this example
        reward = None
        print("Failed to parse gold solution: ", sol)
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
    return results.strip()

def verify_gsm8k(answer_parsed, gold_parsed):
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
            if any([v < 0 for v in results]):
                return {'valid': False}
            return {'dt': results, 'valid': True}
        else:
            return {'valid': False}
    except:
        return {'valid': False}





input_file = "/mnt/bn/ocr-rl-data/whiteboxRL/data/gsm8k_doubao_response.jsonl"
input_data = []

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        input_data.append(data)


dataset = load_dataset("gsm8k", "main", split="train")
# question-answer pair construction
question_answer_pairs = {}
for example in dataset:
    question = example["question"]
    answer = example["answer"]
    question_answer_pairs[question] = answer


evaluation_accuracy = 0
evaluation_format = 0
evaluation_rewrew = 0

for item in input_data:
    # extract question
    raw_data = item["message"]["content"][0]["text"]
    item["question"] = raw_data.split("Your Question:\n\n")[1]
    item["answer"] = question_answer_pairs[item["question"]]
    item["pure_answer"] = question_answer_pairs[item["question"]].split('####')[1].strip()
    item["response"] = item["response"].replace("<thinking>", "<think>").replace("</thinking>", "</think>").replace("<answering>", "<answer>").replace("</answering>", "</answer>")
    # statistic: evaluation accuracy
    item["stat_accuracy_reward"] = accuracy_reward(item["response"], item["pure_answer"])
    item["stat_format_all_reward"] = format_reward_all(item["response"])

    # get rewrew
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
print(f"accuracy: {evaluation_accuracy} / {len(input_data)}")
print(f"format: {evaluation_format} / {len(input_data)}")
print(f"rewrew: {evaluation_rewrew} / {len(input_data)}")


# write to file 
with open(input_file.replace('.jsonl', '_FIN.jsonl'), 'w', encoding='utf-8') as f:
    for item in input_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
