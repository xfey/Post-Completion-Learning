from functools import partial


system_prompt = "You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>\n...\n</think>\n<answer>\n...\n</answer>"


def formatting_gsm8k_sft(example, use_system_prompt=True):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>\n' + reasoning + "\n</think>\n<answer>\n" + result + "\n</answer>"
    else:
        formatted_answer = answer_text
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def formatting_gsm8k_rft(example, use_system_prompt=True):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>\n' + reasoning + "\n</think>\n<answer>\n" + result + "\n</answer>"
    else:
        formatted_answer = answer_text
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "prompt": messages[:-1], "solution": formatted_answer}


def formatting_gsm8k_grpo(example, use_system_prompt=True):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>\n' + reasoning + "\n</think>\n<answer>\n" + result + "\n</answer>"
    else:
        formatted_answer = answer_text
    return {"prompt": messages, "solution": formatted_answer}


def formatting_math220k_grpo(example, use_system_prompt=True):
    messages = []
    if use_system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": example['problem']})
    return {"prompt": messages}


def preprocess(dataset_name, use_system_prompt=True):

    if "gsm8k" in dataset_name:
        return partial(formatting_gsm8k_rft, use_system_prompt=use_system_prompt)
    elif "Math-220k" in dataset_name:
        # GRPO only, not implemented
        return partial(formatting_math220k_grpo, use_system_prompt=use_system_prompt)
    
    # if train_type == "sft":
    #     if "gsm8k" in dataset_name:
    #         return partial(formatting_gsm8k_sft, use_system_prompt=use_system_prompt)
    #     return None
    # elif train_type == "grpo":
    #     if "Math-220k" in dataset_name:
    #         return partial(formatting_math220k_grpo, use_system_prompt=use_system_prompt)
    #     elif "gsm8k" in dataset_name:
    #         return partial(formatting_gsm8k_grpo, use_system_prompt=use_system_prompt)
    #     return None
    # else:
    #     raise ValueError
