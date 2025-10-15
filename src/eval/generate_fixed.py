import os
import json
import argparse
import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def main(model_path, output_file):

    if os.path.exists(output_file):
        print(f"File {output_file} exists, skip generation.")
        return

    device = "cuda:0" # the device to load the model onto

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据集
    dataset = load_dataset("openai/gsm8k", "main", split="test")


    with open(output_file, 'a', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(dataset, desc="Generating responses")):
            question = item['question']
            ground_truth = item['answer']

            messages = [
                {"role": "system", "content": "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"},
                {"role": "user", "content": question}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=1024,
                    do_sample=False
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
    args = parser.parse_args()

    model_path = os.path.join("/mnt/bn/ocr-rl-data/whiteboxRL/experiments/GSM8k_Qwen25_7B_Instruct/", args.exp)
    output_file = os.path.join(args.output_root, args.exp + ".jsonl")

    main(model_path, output_file)
