import json
import os
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
import time


system_prompt_taer = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer, respond in the following format: <thinking>...</thinking><answering>...</answering>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation><reward>...</reward>

Example demonstration:
User:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Assistant:
<thinking>It takes 2/2=<<2/2=1>>1 bolt of white fiber, so the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</thinking>
<answering>3</answering>
<evaluation>The calculation 48/2=24 is right, and 48+24=72 is also correct. My answer uses the required tags properly, so format is correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <thinking>...</thinking><answering>...</answering><evaluation>...</evaluation><reward>...</reward>
"""


system_prompt_taer_mathqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer, respond in the following format: <thinking>...</thinking><answering>...</answering>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation><reward>...</reward>

Example demonstration:
User:
average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\nOptions: a ) 1200 , b ) 120 , c ) 360 , d ) 240 , e ) none of these
Assistant:
<thinking>explanation : let the original no . of students be x . according to situation , 40 x + 120 * 32 = ( x + 120 ) 36 ⇒ x = 120 so , required no . of students after joining the new students = x + 120 = 240 . answer : d</thinking>
<answering>d</answering>
<evaluation>My formula can be simplified to be: 40 x = 36 x + 120 * 4, thus 4 x = 4 * 120 ⇒ x = 120, so I choose option d, my answer is right. My answer uses the required tags properly, so format is also correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <thinking>...</thinking><answering>...</answering><evaluation>...</evaluation><reward>...</reward>
"""


class Caller:
    def __init__(self, api_key=""):
        self.client = OpenAI(
            base_url="https://ark-cn-beijing.bytedance.net/api/v3",
            api_key=api_key,
        )
    
    def get_response(self, question, options):
        """获取大模型的回复"""
        try:
            # 组合问题和选项
            user_input = f"{question}\nOptions: {options}"
            
            response = self.client.chat.completions.create(
                model="ep-20250715001055-tmhwv",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt_taer_mathqa}]
                    },
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": user_input}]
                    }
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {e}")
            return f"Error: {str(e)}"


def process_mathqa_dataset(api_key="", output_dir="./processed_mathqa"):
    """处理MathQA数据集"""
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据集
    print("正在加载MathQA数据集...")
    ds = load_dataset("allenai/math_qa", trust_remote_code=True)
    
    # 初始化API调用器
    caller = Caller(api_key) if api_key else None
    
    processed_data = []
    # 处理train和validation数据集
    for split_name in ['train', 'validation']:
        print(f"\n正在处理 {split_name} 数据集...")
        
        split_data = ds[split_name]
        
        # 处理每个样本
        for i, sample in enumerate(tqdm(split_data, desc=f"Processing {split_name}")):
            # 保留原有字段
            processed_sample = {
                'custom_id': f"{split_name}_{str(i).zfill(6)}",
                "body": {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt_taer_mathqa,
                        },
                        {
                            "role": "user", 
                            "content": f"{sample['Problem']}\nOptions: {sample['options']}",
                        }
                    ],
                },
                "max_tokens": 4096
            }

            processed_data.append(json.dumps(processed_sample, ensure_ascii=False))
            
        
    # 保存最终处理结果
    output_file = os.path.join(output_dir, "doubao_offline_trainval.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for pd in processed_data:
            f.write(pd+'\n')
        
    print(f"{split_name} 数据集处理完成，共 {len(processed_data)} 个样本")
    print(f"保存到: {output_file}")


def load_processed_data(file_path):
    """加载处理后的数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def sample_analysis(data_file):
    """分析处理后的数据样本"""
    data = load_processed_data(data_file)
    
    print(f"数据集大小: {len(data)}")
    print(f"字段: {list(data[0].keys())}")
    print("\n第一个样本:")
    for key, value in data[0].items():
        if key == 'response' and len(str(value)) > 200:
            print(f"{key}: {str(value)[:200]}...")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    # 设置你的API key
    API_KEY = "f9bd0aca-a0ca-4f7a-9f74-702d6322cd9b"  # 请填入你的API key
    
    # 处理数据集
    # 如果不想调用API，可以将API_KEY设为空字符串
    process_mathqa_dataset(api_key=API_KEY, output_dir="/mnt/bn/ocr-rl-data/whiteboxRL/data/mathqa_doubao_response")
