import json
from copy import deepcopy
import re





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
    if results.lower() == 'true':
        return True
    elif results.lower() == 'false':
        return False


def verify_gsm8k(answer_parsed, gold_parsed):
    # if gold_parsed.lower() == 'true':
    #     gold_parsed = True
    # elif gold_parsed.lower() == 'false':
    #     gold_parsed = False
    
    if answer_parsed == gold_parsed:
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


res_file = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_doubao_response/strategyqa.json", 'r'))
ori_file = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/train.json", 'r'))

new_res = []

for i in range(len(res_file)):
    assert res_file[i]['qid'] == ori_file[i]['qid']
    r = deepcopy(res_file[i])
    r['facts'] = ori_file[i]['facts']
    answer = r['answer']

    r["stat_accuracy_reward"] = accuracy_reward(r["response"], answer)
    r["stat_format_all_reward"] = format_reward_all(r["response"])
    rewrew = rewarding_reward_raw(r["response"])

    if rewrew['valid'] == False:
        r["stat_rewarding_reward"] = []
    else:
        r["stat_rewarding_reward"] = rewrew['dt']

    if r["stat_rewarding_reward"] and len(r["stat_rewarding_reward"]) == 2:
        if r["stat_rewarding_reward"][0] == r["stat_accuracy_reward"] and r["stat_rewarding_reward"][1] == r["stat_format_all_reward"]:
            r["stat_rewrew_right"] = True
        else:
            r["stat_rewrew_right"] = False
    r["stat_rewrew_right"] = r.get("stat_rewrew_right", False)


    evaluation_accuracy += r["stat_accuracy_reward"]
    evaluation_format += r["stat_format_all_reward"]
    evaluation_rewrew += int(r["stat_rewrew_right"])
    
    new_res.append(r)



# print stat results
print(f"accuracy: {evaluation_accuracy} / {len(res_file)}")
print(f"format: {evaluation_format} / {len(res_file)}")
print(f"rewrew: {evaluation_rewrew} / {len(res_file)}")



json.dump(new_res, open("/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_doubao_response/strategyqa_FIN.json", 'w'), indent=2)
