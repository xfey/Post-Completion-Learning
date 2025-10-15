import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList



system_prompt_ta = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>...</think>\n<answer>...</answer>

Example demonstration:
User:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Assistant:
<think>It takes 2/2=<<2/2=1>>1 bolt of white fiber, so the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</think>
<answer>3</answer>
---
Your FINAL response should follow the format: <think>...</think><answer>...</answer>
"""


system_prompt_taer = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer, respond in the following format: <think>...</think><answer>...</answer>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation><reward>...</reward>

Example demonstration:
User:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Assistant:
<think>It takes 2/2=<<2/2=1>>1 bolt of white fiber, so the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</think>
<answer>3</answer>
<evaluation>The calculation 48/2=24 is right, and 48+24=72 is also correct. My answer uses the required tags properly, so format is correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <think>...</think><answer>...</answer><eval>...</eval><reward>...</reward>
"""


system_prompt_taer_strategyqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer (True or False), respond in the following format: <think>...</think>\n<answer>...</answer>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation>\n<reward>...</reward>

Example demonstration:
User:
Can you buy Casio products at Petco?
Assistant:
<think>Casio is a manufacturer of consumer electronics and watches. Petco is a chain store that sells pet supplies like food, bowls, litter, toys, cages and grooming equipment.</think>
<answer>False</answer>
<evaluation>Let me recheck the facts. Casio products are consumer electronics and watches is right, and Petco does sell pet goods. So the electronics and watches are not included in pet stores, and the result is False. My answer is correct. My answer uses the required tags properly, so format is also correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>\n<evaluation>...</evaluation>\n<reward>...</reward>
"""


system_prompt_ta_strategyqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer (True or False), respond in the following format: <think>...</think>\n<answer>...</answer>

Example demonstration:
User:
Can you buy Casio products at Petco?
Assistant:
<think>Casio is a manufacturer of consumer electronics and watches. Petco is a chain store that sells pet supplies like food, bowls, litter, toys, cages and grooming equipment.</think>
<answer>False</answer>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>
"""





class EndOfAnswerStoppingCriteria(StoppingCriteria):
    def __init__(self, tokenizer, stop_string="</answer>"):
        self.tokenizer = tokenizer
        self.stop_string = stop_string
        self.stop_string_length = len(stop_string)
        
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取最后生成的部分文本
        last_tokens = input_ids[0][-self.stop_string_length*2:]  # 取足够长的token序列
        if len(last_tokens) < self.stop_string_length:
            return False
            
        # 解码最后生成的文本
        decoded_text = self.tokenizer.decode(last_tokens, skip_special_tokens=True)
        
        # 检查是否包含停止字符串
        return self.stop_string in decoded_text




def main(model_path, output_file, use_stopping_criteria=False):

    if os.path.exists(output_file):
        print(f"File {output_file} exists, skip generation.")
        return

    device = "cuda" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 创建停止条件（可选）
    stopping_criteria = None
    if use_stopping_criteria:
        stopping_criteria = StoppingCriteriaList([EndOfAnswerStoppingCriteria(tokenizer)])

    # 加载数据集
    # dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = load_dataset("json", data_files=["/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_github/dev_formatted.json"])['train'] # default to `train` split


    with open(output_file, 'a', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(dataset, desc="Generating responses")):
            question = item['question']
            ground_truth = item['answer']

            messages = [
                {"role": "system", "content": system_prompt_taer_strategyqa},
                {"role": "user", "content": question}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                generation_kwargs = {
                    "max_new_tokens": 2048,
                    "do_sample": False
                }
                if stopping_criteria is not None:
                    generation_kwargs["stopping_criteria"] = stopping_criteria
                    
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    **generation_kwargs
                )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            ## write
            result = {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "generated_result": response,
                "prompt": text
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()  # 确保及时写入

if __name__ == "__main__":
    """
        model_path = "/mnt/bn/ocr-rl-data/model/Qwen2.5-7B-Instruct/"
        output_file = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/baseline_ckpt.jsonl"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/")
    parser.add_argument("--exp_series", type=str, default="1")
    parser.add_argument("--early_stop", action="store_true", help="Use custom stopping criteria to stop generation at </answer>")
    args = parser.parse_args()

    if args.exp_series == "1":
        model_root = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/GSM8k_Qwen25_7B_Instruct/"
    elif args.exp_series == "2":
        model_root = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/GSM8k_Qwen25_1_5B_Instruct/"
    elif args.exp_series == "3":
        model_root = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/GSM8k_LLAMA/"

    model_path = os.path.join(model_root, args.exp)
    output_file = os.path.join(args.output_root, args.exp + ".jsonl")

    main(model_path, output_file, args.early_stop)
