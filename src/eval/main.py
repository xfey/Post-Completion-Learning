import os
import json
import argparse
from typing import List, Optional
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


def extract_answer_from_response(response: str) -> str:
    """从模型响应中提取答案部分"""
    # 提取 <answer> 标签内的内容
    start_tag = "<answer>"
    end_tag = "</answer>"
    
    start_idx = response.find(start_tag)
    if start_idx == -1:
        return ""  # 如果没有找到标签，返回整个响应
    
    start_idx += len(start_tag)
    end_idx = response.find(end_tag, start_idx)
    if end_idx == -1:
        return ""
    
    return response[start_idx:end_idx].strip()


def evaluate_accuracy(generated_results: List[str], ground_truth: List[str]) -> List[Optional[float]]:
    """评估生成结果的准确率，基于 reward.py 中的 accuracy_reward 函数"""
    rewards = []
    accuracy = []
    
    for generated, truth in zip(generated_results, ground_truth):
        # 解析真实答案
        gold_parsed = parse(truth, extraction_mode="first_match")
        
        if len(gold_parsed) == 0:
            # 如果真实答案无法解析，跳过该样本
            rewards.append(0.0)
            accuracy.append(0.0)
            print(f"Failed to parse ground truth: {truth}")
            continue
        
        # 从生成结果中提取答案部分
        answer_content = extract_answer_from_response(generated)
        # answer_content = generated
        
        # 解析生成的答案
        answer_parsed = parse(answer_content, extraction_mode="first_match")
        try:
            reward = float(verify(gold_parsed, answer_parsed))
            accuracy.append( answer_content.strip() == truth.strip() )
        except Exception as e:
            print(f"Verification failed: {e}, generated: {answer_parsed}, ground_truth: {gold_parsed}")
            reward = 0.0
            accuracy.append(0.0)
        
        rewards.append(reward)
    
    return rewards, accuracy


def main():
    # parser = argparse.ArgumentParser(description="评估 GSM8k 数据集上的模型准确率")
    # parser.add_argument("--input_file", type=str, required=True, help="包含生成结果的 JSONL 文件路径")
    # parser.add_argument("--dataset_split", type=str, default="test", help="数据集分割 (默认: test)")
    # args = parser.parse_args()
    
    # 加载生成的结果
    # print(f"Loading generated results from {args.input_file}")
    # 加载 GSM8k 数据集
    # print(f"Loading GSM8k dataset ({args.dataset_split} split)")
    dataset = load_dataset("openai/gsm8k", "main", split="test")


    all_files = os.listdir("/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/")
    for file in all_files:
        input_file = os.path.join("/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/generate/", file)
        print(f"Loading generated results from {input_file}")
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
            exit(0)
        
        # 提取真实答案
        ground_truth = [item['answer'] for item in dataset]
        ground_truth = [item.split('####')[1] for item in ground_truth]
        
        # 评估准确率
        print("Evaluating accuracy...")
        rewards, accuracy = evaluate_accuracy(generated_results, ground_truth)
        
        # print(f"Rewards average: {sum(rewards) / len(rewards)}")
        # print(f"Accuracy average: {sum(accuracy) / len(accuracy)}")

        ave_reward = sum(rewards) / len(rewards) * 100
        ave_acc = sum(accuracy) / len(accuracy) * 100

        print(f"acc(parse) = {ave_reward:.2f}, acc(full_match) = {ave_acc:.2f}")

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
