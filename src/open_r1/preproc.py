from functools import partial


system_prompt_ta = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>...</think>\n<answer>...</answer>

Example demonstration:
User:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Assistant:
<think>It takes 2/2=<<2/2=1>>1 bolt of white fiber, so the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</think>
<answer>3</answer>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>
"""


system_prompt_taer = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer, respond in the following format: <think>...</think>\n<answer>...</answer>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation>\n<reward>...</reward>

Example demonstration:
User:
A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?
Assistant:
<think>It takes 2/2=<<2/2=1>>1 bolt of white fiber, so the total amount of fabric is 2+1=<<2+1=3>>3 bolts of fabric</think>
<answer>3</answer>
<evaluation>The calculation 48/2=24 is right, and 48+24=72 is also correct. My answer uses the required tags properly, so format is correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>\n<evaluation>...</evaluation>\n<reward>...</reward>
"""


### ----------------------


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


### ----------------------



system_prompt_taer_mathqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. 
1. You first think about the reasoning process as an internal monologue and then provide the user with the answer, respond in the following format: <think>...</think>\n<answer>...</answer>
2. You then evaluate your solution with the following reward functions:
  - accuracy reward: 1 point for correct answer, 0 for incorrect
  - format reward: 1 point for proper think&answer tags, 0 for improper
Respond your evaluation and reward scores [accuracy reward, format reward] in the following format: <evaluation>...</evaluation>\n<reward>...</reward>

Example demonstration:
User:
average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\nOptions: a ) 1200 , b ) 120 , c ) 360 , d ) 240 , e ) none of these
Assistant:
<think>explanation : let the original no . of students be x . according to situation , 40 x + 120 * 32 = ( x + 120 ) 36 ⇒ x = 120 so , required no . of students after joining the new students = x + 120 = 240 . answer : d</think>
<answer>d</answer>
<evaluation>My formula can be simplified to be: 40 x = 36 x + 120 * 4, thus 4 x = 4 * 120 ⇒ x = 120, so I choose option d, my answer is right. My answer uses the required tags properly, so format is also correct.</evaluation>
<reward>[1,1]</reward>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>\n<evaluation>...</evaluation>\n<reward>...</reward>
"""


system_prompt_ta_mathqa = """You are a helpful AI Assistant that provides well-reasoned and detailed responses. You first think about the reasoning process as an internal monologue and then provide the user with the answer. Respond in the following format: <think>...</think>\n<answer>...</answer>

Example demonstration:
User:
average age of students of an adult school is 40 years . 120 new students whose average age is 32 years joined the school . as a result the average age is decreased by 4 years . find the number of students of the school after joining of the new students .\nOptions: a ) 1200 , b ) 120 , c ) 360 , d ) 240 , e ) none of these
Assistant:
<think>explanation : let the original no . of students be x . according to situation , 40 x + 120 * 32 = ( x + 120 ) 36 ⇒ x = 120 so , required no . of students after joining the new students = x + 120 = 240 . answer : d</think>
<answer>d</answer>
---
Your FINAL response should follow the format: <think>...</think>\n<answer>...</answer>
"""




def formatting_gsm8k_sft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_ta})
    # messages.append({"role": "user", "content": sft_icl_question_instruction + example["question"]})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>' + reasoning + "</think>\n<answer>" + result + "</answer>"
    else:
        formatted_answer = answer_text
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}


# def formatting_gsm8k_rft(example, use_system_prompt=True):
def formatting_gsm8k_rft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer})
    # messages.append({"role": "user", "content": question_instruction + example["question"]})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>' + reasoning + "</think>\n<answer>" + result + "</answer>"
    else:
        formatted_answer = answer_text
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "prompt": messages[:-1], "solution": formatted_answer}



def formatting_gsm8k_custom_data_sft(example, using_prompt=None):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer})
    # messages.append({"role": "user", "content": question_instruction + example["question"]})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["response"]
    messages.append({"role": "assistant", "content": answer_text})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def formatting_gsm8k_custom_data_rft(example, using_prompt="taer"):
    messages = []
    if using_prompt == "taer":
        # prompt = question_instruction
        messages.append({"role": "system", "content": system_prompt_taer})
    elif using_prompt == "ta":
        # prompt = sft_icl_question_instruction
        messages.append({"role": "system", "content": system_prompt_ta})
    # messages.append({"role": "user", "content": prompt + example["question"]})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    if "####" in answer_text:
        reasoning, result = answer_text.split("####")
        result = result.strip()
        reasoning = reasoning.strip()
        formatted_answer = '<think>' + reasoning + "</think>\n<answer>" + result + "</answer>"
    else:
        formatted_answer = answer_text
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "rewrew_valid": example["stat_rewrew_right"],
            "taer_response": example["response"],
            "prompt": messages[:-1], "solution": formatted_answer}


### ----------------------



def formatting_strategyqa_sft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_ta_strategyqa})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = "True" if example["answer"] else "False"

    formatted_answer = "<think>" + ' '.join(example["facts"]) + "</think>\n<answer>" + answer_text + "</answer>"

    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def formatting_strategyqa_rft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer_strategyqa})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = "True" if example["answer"] else "False"
    
    formatted_answer = "<think>" + ' '.join(example["facts"]) + "</think>\n<answer>" + answer_text + "</answer>"
    
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "prompt": messages[:-1], "solution": formatted_answer}



def formatting_strategyqa_custom_data_sft(example, using_prompt=None):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer_strategyqa})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["response"]
    messages.append({"role": "assistant", "content": answer_text})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}



def formatting_strategyqa_custom_data_rft(example, using_prompt="taer"):
    messages = []
    if using_prompt == "taer":
        # prompt = question_instruction
        messages.append({"role": "system", "content": system_prompt_taer_strategyqa})
    elif using_prompt == "ta":
        # prompt = sft_icl_question_instruction
        messages.append({"role": "system", "content": system_prompt_ta_strategyqa})

    messages.append({"role": "user", "content": example["question"]})
    answer_text = "True" if example["answer"] else "False"

    formatted_answer = "<think>" + ' '.join(example["facts"]) + "</think>\n<answer>" + answer_text + "</answer>"

    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "rewrew_valid": example["stat_rewrew_right"],
            "taer_response": example["response"],
            "prompt": messages[:-1], "solution": formatted_answer}



### ----------------------



def formatting_mathqa_sft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_ta_mathqa})
    messages.append({"role": "user", "content": example["question"]})   # with options already
    answer_text = example["answer"]

    formatted_answer = "<think>" + example["rationale"] + "</think>\n<answer>" + answer_text + "</answer>"

    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}



def formatting_mathqa_rft(example):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer_mathqa})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]
    
    formatted_answer = "<think>" + example["rationale"] + "</think>\n<answer>" + answer_text + "</answer>"
    
    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "prompt": messages[:-1], "solution": formatted_answer}



def formatting_mathqa_custom_data_sft(example, using_prompt=None):
    messages = []
    # if use_system_prompt:
    messages.append({"role": "system", "content": system_prompt_taer_mathqa})
    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["response"]
    messages.append({"role": "assistant", "content": answer_text})
    # return {"sft_messages": messages}
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def formatting_mathqa_custom_data_rft(example, using_prompt="taer"):
    messages = []
    if using_prompt == "taer":
        # prompt = question_instruction
        messages.append({"role": "system", "content": system_prompt_taer_mathqa})
    elif using_prompt == "ta":
        # prompt = sft_icl_question_instruction
        messages.append({"role": "system", "content": system_prompt_ta_mathqa})

    messages.append({"role": "user", "content": example["question"]})
    answer_text = example["answer"]

    formatted_answer = "<think>" + example["rationale"] + "</think>\n<answer>" + answer_text + "</answer>"

    messages.append({"role": "assistant", "content": formatted_answer})
    # return {"sft_messages": messages,
    return {"sft_completion": messages[-1:],
            "rewrew_valid": example["stat_rewrew_right"],
            "taer_response": example["response"],
            "prompt": messages[:-1], "solution": formatted_answer}





# def formatting_math220k_grpo(example, use_system_prompt=True):
#     messages = []
#     if use_system_prompt:
#         messages.append({"role": "system", "content": system_prompt})
#     messages.append({"role": "user", "content": example['problem']})
#     return {"prompt": messages}


def preprocess(dataset_name, use_system_prompt=True, datatype="rft", using_prompt="taer"):
    if "gsm8k_" in dataset_name:
        # 自己刷的SFT数据集
        if datatype == "rft":
            # return partial(formatting_gsm8k_rft, use_system_prompt=use_system_prompt)
            return partial(formatting_gsm8k_custom_data_rft, using_prompt=using_prompt)
        elif datatype == "sft":
            return partial(formatting_gsm8k_custom_data_sft, using_prompt=using_prompt)
    elif "gsm8k" in dataset_name:
        if datatype == "rft":
            # return partial(formatting_gsm8k_rft, use_system_prompt=use_system_prompt)
            return partial(formatting_gsm8k_rft)
        elif datatype == "sft":
            return partial(formatting_gsm8k_sft)
    
    ## StrategyQA
    if "strategyqa_" in dataset_name:
        if datatype == "rft":
            return partial(formatting_strategyqa_custom_data_rft, using_prompt=using_prompt)
        elif datatype == "sft":
            return partial(formatting_strategyqa_custom_data_sft, using_prompt=using_prompt)
    elif "strategyqa" in dataset_name:
        if datatype == "rft":
            raise NotImplementedError
            # return partial(formatting_strategyqa_rft)
        elif datatype == "sft":
            return partial(formatting_strategyqa_sft)

    ## MathQA
    if "mathqa_" in dataset_name:
        if datatype == "rft":
            return partial(formatting_mathqa_custom_data_rft, using_prompt=using_prompt)
        elif datatype == "sft":
            return partial(formatting_mathqa_custom_data_sft, using_prompt=using_prompt)
    elif "mathqa" in dataset_name:
        if datatype == "rft":
            raise NotImplementedError
            # return partial(formatting_mathqa_rft)
        elif datatype == "sft":
            return partial(formatting_mathqa_sft)



    elif "Math-220k" in dataset_name:
        # GRPO only, not implemented
        # return partial(formatting_math220k_grpo, use_system_prompt=use_system_prompt)
        raise NotImplementedError
