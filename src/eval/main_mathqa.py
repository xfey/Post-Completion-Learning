import os
import re
import json
import argparse
from typing import List, Optional
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def accuracy_reward(contents: list[str], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth.""" 
    rewards = []
    for content, sol in zip(contents, solution):
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
        rewards.append(reward)
    return rewards

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
    # if gold_parsed.lower() == 'true':
    #     gold_parsed = True
    # elif gold_parsed.lower() == 'false':
    #     gold_parsed = False
    
    if answer_parsed == gold_parsed:
        return True
    return False


def main():
    # parser = argparse.ArgumentParser(description="评估 GSM8k 数据集上的模型准确率")
    # parser.add_argument("--input_file", type=str, required=True, help="包含生成结果的 JSONL 文件路径")
    # parser.add_argument("--dataset_split", type=str, default="test", help="数据集分割 (默认: test)")
    # args = parser.parse_args()
    
    # 加载生成的结果
    # print(f"Loading generated results from {args.input_file}")
    # 加载 GSM8k 数据集
    # print(f"Loading GSM8k dataset ({args.dataset_split} split)")
    # dataset = load_dataset("openai/gsm8k", "main", split="test")
    dataset = load_dataset("json", data_files=["/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response/raw_mathqa_test.json"])['train'] # default to `train` split

    all_files = os.listdir("/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/")
    for file in all_files:
        if not (file.startswith('exp12') or file.startswith('exp13')):
            continue
        input_file = os.path.join("/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/", file)
        print(f"Loading generated results from {os.path.basename(input_file)}")
        generated_results = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                generated_results.append(data['generated_result'])
        
        
        # 确保数据长度匹配
        if len(generated_results) != len(dataset):
            # print(f"Warning: Generated results length ({len(generated_results)}) != dataset length ({len(dataset)})")
            # min_len = min(len(generated_results), len(dataset))
            # generated_results = generated_results[:min_len]
            # dataset = dataset.select(range(min_len))
            print(f'evaluation result imcomplete: {len(generated_results)} / {len(dataset)}')
            continue
        
        # 提取真实答案
        ground_truth = [item['correct'].strip().lower() for item in dataset]
        # ground_truth = [item.split('####')[1] for item in ground_truth]
        
        # 评估准确率
        # print("Evaluating accuracy...")
        rewards = accuracy_reward(generated_results, ground_truth)
        
        # print(f"Rewards average: {sum(rewards) / len(rewards)}")
        # print(f"Accuracy average: {sum(accuracy) / len(accuracy)}")

        ave_reward = sum(rewards) / len(rewards) * 100
        # ave_acc = sum(accuracy) / len(accuracy) * 100

        print(f"acc = {ave_reward:.2f}")
        # print(f"acc(parse) = {ave_reward:.2f}, acc(full_match) = {ave_acc:.2f}")

        print("="*30)

    # # 计算统计信息
    # valid_rewards = [r for r in rewards if r is not None]
    # if valid_rewards:
    #     accuracy = sum(valid_rewards) / len(valid_rewards)
    #     print(f"\nResults:")
    #     print(f"Total samples: {len(rewards)}")
    #     print(f"Valid samples: {len(valid_rewards)}")
    #     print(f"Skipped samples: {len(rewards) - len(valid_rewards)}")
    #     print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    # else:
    #     print("No valid samples found!")
    
    # # 输出详细结果
    # print(f"\nDetailed results:")
    # correct_count = sum(1 for r in valid_rewards if r == 1.0)
    # print(f"Correct answers: {correct_count}")
    # print(f"Incorrect answers: {len(valid_rewards) - correct_count}")


if __name__ == "__main__":
    main()
