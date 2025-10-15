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


system_prompt_taer_strategyqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer (True or False), respond in the following format: <thinking>...</thinking><answering>...</answering>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation><reward>...</reward>

Example demonstration:
User:
Can you buy Casio products at Petco?
Assistant:
<thinking>Casio is a manufacturer of consumer electronics and watches. Petco is a chain store that sells pet supplies like food, bowls, litter, toys, cages and grooming equipment.</thinking>
<answering>False</answering>
<evaluation>Let me recheck the facts. Casio products are consumer electronics and watches is right, and Petco does sell pet goods. So the electronics and watches are not included in pet stores, and the result is False. My answer is correct. My answer uses the required tags properly, so format is also correct.</evaluation>
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
    
    def get_response(self, question):
        """获取大模型的回复"""
        try:
            # 组合问题和选项
            user_input = f"{question}"
            
            response = self.client.chat.completions.create(
                model="ep-20250715001055-tmhwv",
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": system_prompt_taer_strategyqa}]
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
    # print("正在加载MathQA数据集...")
    # ds = load_dataset("allenai/math_qa", trust_remote_code=True)
    
    # ds = json.load(open("/mnt/bn/ocr-rl-data/datasets/StrategyQA/train.json", "r"))
    ds = json.load(open("/mnt/bn/ocr-rl-data/whiteboxRL/code/open-r1-whitebox/strategyqa_github/train.json", "r"))
    
    # 初始化API调用器
    caller = Caller(api_key) if api_key else None
    
    processed_data = []
    
    # 处理每个样本
    for i, sample in enumerate(tqdm(ds, desc=f"Processing")):
        # 保留原有字段
        processed_sample = {
            'qid': sample['qid'],
            'question': sample['question'],
            'answer': sample['answer'],
        }
        
        # # 添加query字段（问题+选项的组合）
        # processed_sample['query'] = f"{sample['question']}"
        
        # 添加response字段（如果提供了API key则调用大模型）
        if caller:
            try:
                response = caller.get_response(sample['question'])
                processed_sample['response'] = response
                print(response)
                
                # 添加短暂延迟避免API限流
                time.sleep(0.05)
                
            except Exception as e:
                processed_sample['response'] = f"Error: {str(e)}"
                print(f"处理第{i}个样本时出错: {e}")
        else:
            # 如果没有提供API key，response字段为空
            processed_sample['response'] = ""
        
        processed_data.append(processed_sample)
        
        # 每处理100个样本保存一次（防止数据丢失）
        if (i + 1) % 10 == 0:
            temp_file = os.path.join(output_dir, f"strategyqa_temp.json")
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 保存最终处理结果
    output_file = os.path.join(output_dir, f"strategyqa.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    print(f"数据集处理完成，共 {len(processed_data)} 个样本")
    print(f"保存到: {output_file}")
    
    # 删除临时文件
    temp_file = os.path.join(output_dir, f"strategyqa_temp.json")
    if os.path.exists(temp_file):
        os.remove(temp_file)
    
    print(f"\n所有数据处理完成，文件保存在: {output_dir}")


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
    process_mathqa_dataset(api_key=API_KEY, output_dir="/mnt/bn/ocr-rl-data/whiteboxRL/data/strategyqa_doubao_response")
    
    # 分析处理结果（可选）
    # sample_analysis("./processed_mathqa/train.json")