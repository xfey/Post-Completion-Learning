import json
import argparse
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# 配置参数
model_path = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/GSM8k_Qwen25_7B_Instruct/exp006"
output_file = "/mnt/bn/ocr-rl-data/whiteboxRL/experiments/evaluation/exp006.jsonl"

# 初始化 vLLM 模型
llm = LLM(
    model=model_path,
    tensor_parallel_size=4,  # 根据GPU数量调整
    trust_remote_code=True,
    dtype="auto",
    max_model_len=4096,  # 根据需要调整
    gpu_memory_utilization=0.9,  # GPU内存使用率
    max_num_seqs=64,  # 批处理大小，根据GPU内存调整
)

# 初始化 tokenizer（用于格式化消息）
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 设置采样参数
sampling_params = SamplingParams(
    temperature=0.0,  # do_sample=False 对应 temperature=0
    max_tokens=1024,
    top_p=1.0,
    stop=None,
    skip_special_tokens=True
)

# 加载数据集
dataset = load_dataset("openai/gsm8k", "main", split="test")

def format_messages(question):
    """格式化消息为 chat template"""
    messages = [
        {"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"},
        {"role": "user", "content": question}
    ]
    
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

def process_batch(batch_data, batch_size=32):
    """批处理数据"""
    results = []
    
    for i in range(0, len(batch_data), batch_size):
        batch = batch_data[i:i + batch_size]
        
        # 准备批次数据
        prompts = []
        batch_info = []
        
        for idx, item in batch:
            question = item['question']
            ground_truth = item['answer']
            
            prompt = format_messages(question)
            prompts.append(prompt)
            batch_info.append({
                "index": idx,
                "question": question,
                "ground_truth": ground_truth,
                "prompt": prompt
            })
        
        # 批量推理
        outputs = llm.generate(prompts, sampling_params)
        
        # 处理结果
        for info, output in zip(batch_info, outputs):
            response = output.outputs[0].text
            result = {
                "index": info["index"],
                "question": info["question"],
                "ground_truth": info["ground_truth"],
                "generated_result": response,
                "prompt": info["prompt"]
            }
            results.append(result)
    
    return results

# 方法1：批处理推理（推荐）
def batch_inference():
    """批处理推理方式"""
    # 准备所有数据
    all_data = [(i, item) for i, item in enumerate(dataset)]
    
    # 批处理推理
    results = process_batch(all_data, batch_size=32)  # 根据GPU内存调整batch_size
    
    # 写入结果
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in tqdm(results, desc="Writing results"):
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

# 方法2：流式推理（内存友好）
def streaming_inference():
    """流式推理方式，适合大数据集"""
    batch_size = 32
    batch_data = []
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(dataset, desc="Processing")):
            batch_data.append((i, item))
            
            # 当批次满了或者是最后一批时，进行推理
            if len(batch_data) == batch_size or i == len(dataset) - 1:
                results = process_batch(batch_data, batch_size=batch_size)
                
                # 写入结果
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                f.flush()
                
                batch_data = []

# 方法3：单条推理（与原代码最相似）
def single_inference():
    """单条推理方式"""
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(dataset, desc="Generating responses")):
            question = item['question']
            ground_truth = item['answer']
            
            prompt = format_messages(question)
            
            # 单条推理
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text
            
            result = {
                "index": i,
                "question": question,
                "ground_truth": ground_truth,
                "generated_result": response,
                "prompt": prompt
            }
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "streaming", "single"], 
                       default="streaming", help="推理模式")
    parser.add_argument("--batch_size", type=int, default=32, 
                       help="批处理大小")
    
    args = parser.parse_args()
    
    # 根据参数选择推理方式
    if args.mode == "batch":
        print("使用批处理推理...")
        batch_inference()
    elif args.mode == "streaming":
        print("使用流式推理...")
        streaming_inference()
    else:
        print("使用单条推理...")
        single_inference()
    
    print(f"推理完成，结果保存到: {output_file}")
