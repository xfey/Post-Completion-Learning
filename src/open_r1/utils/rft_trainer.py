import torch
from typing import Union, Optional, Any, Dict
from datasets import Dataset, IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl import GRPOConfig

from open_r1.utils.grpo_trainer import GRPOTrainer
from open_r1.reward_evaluator import RewardEvaluator


def pad_sample_to_length(sample, target_length, pad_token_id):
    """
    将单个样本padding到指定长度
    
    Args:
        sample: 包含input_ids, attention_mask, labels的字典
        target_length: 目标长度
        pad_token_id: padding token的id
    
    Returns:
        padding后的样本
    """
    current_length = len(sample["input_ids"])
    if current_length >= target_length:
        # 如果当前长度已经>=目标长度，直接返回（可能需要截断）
        return {
            "input_ids": sample["input_ids"][:target_length],
            "attention_mask": sample["attention_mask"][:target_length],
            "labels": sample["labels"][:target_length]
        }
    
    # 计算需要padding的长度
    padding_needed = target_length - current_length
    padded_input_ids = torch.cat([
        sample["input_ids"], 
        torch.full((padding_needed,), pad_token_id, dtype=sample["input_ids"].dtype)
    ])
    padded_attention_mask = torch.cat([
        sample["attention_mask"],
        torch.zeros(padding_needed, dtype=sample["attention_mask"].dtype)
    ])
    padded_labels = torch.cat([
        sample["labels"],
        torch.full((padding_needed,), -100, dtype=sample["labels"].dtype)
    ])
    
    return {
        "input_ids": padded_input_ids,
        "attention_mask": padded_attention_mask,
        "labels": padded_labels
    }


class RFTTrainer(GRPOTrainer):
    """
    简单的混合 GRPO+SFT 训练器
    对每个样本同时计算 GRPO loss 和 SFT loss，然后相加进行优化
    """
    
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        rft_loss_args: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            **kwargs
        )
        self.reward_function_names = [func.__name__.rstrip('_reward') for func in reward_funcs]
        self.reward_evaluator = RewardEvaluator(self.reward_funcs)

        self.stage1_grpo_weight = rft_loss_args.get("stage1_grpo_weight", 1.0)
        self.stage1_sft_weight = rft_loss_args.get("stage1_sft_weight", 0.1)
        self.stage2_sft_weight = rft_loss_args.get("stage2_sft_weight", 0.1)

        # self._last_total_loss = 0.0

    def _prepare_inputs(self, generation_batch):
        # 保存当前样本数据用于SFT损失计算，由于batch中都是相同样本，取第一个即可
        self._current_sample = generation_batch[0]
        grpo_inputs = super()._prepare_inputs(generation_batch)
        return grpo_inputs

    def prepare_single_sft_sample(self, messages, full_messages):
        prompt_text = self.processing_class.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # 添加assistant提示符
        )
        full_text = self.processing_class.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False
        )
        prompt_ids = self.processing_class.encode(prompt_text, add_special_tokens=False)
        full_ids = self.processing_class.encode(full_text, add_special_tokens=False)
        # 创建labels：只对completion部分计算loss
        labels = [-100] * len(full_ids)
        completion_start = len(prompt_ids)
        if completion_start < len(full_ids):
            labels[completion_start:] = full_ids[completion_start:]
        # 创建attention mask
        attention_mask = [1] * len(full_ids)
        return (full_ids, attention_mask, labels)


    def prepare_stage1_sft_inputs(self):
        """
        从样本数据中准备SFT训练输入。样本数据就是 GRPO 用的问题+答案。
        """
        # 获取对话数据
        messages = []
    
        data_prompt = self._current_sample["prompt"]
        for data_prompt_item in data_prompt:
            if data_prompt_item["role"] == "system":
                messages.append({
                    "role": "system", 
                    "content": data_prompt_item["content"]
                })
            elif data_prompt_item["role"] == "user":
                messages.append({
                    "role": "user", 
                    "content": data_prompt_item["content"]
                })
        
        full_messages = messages + [{
            "role": "assistant", 
            "content": self._current_sample["sft_completion"][0]['content']
        }]

        full_ids, attention_mask, labels = self.prepare_single_sft_sample(messages, full_messages)
        return {
            "input_ids": torch.tensor([full_ids]),
            "attention_mask": torch.tensor([attention_mask]),
            "labels": torch.tensor([labels])
        }


    def prepare_stage2_sft_inputs(self, inputs_data):
        # 计算整个 batch 的结果
        prompt_ids = inputs_data['prompt_ids']
        completion_ids = inputs_data['completion_ids']
        batch_size = prompt_ids.shape[0]
        pad_token_id = self.processing_class.pad_token_id
        
        # 构造评估请求
        reward_functions_str = " | ".join(self.reward_function_names)
        evaluate_request = f"<evaluate> Reward functions: {reward_functions_str}</evaluate>"
        evaluation_data = []

        # 每条数据分别进行评估
        for i in range(batch_size):
            messages = []
            # 从原始数据获得 prompt，正确解析 system + user
            data_prompt = self._current_sample["prompt"]
            for data_prompt_item in data_prompt:
                if data_prompt_item["role"] == "system":
                    messages.append({
                        "role": "system", 
                        "content": data_prompt_item["content"]
                    })
                elif data_prompt_item["role"] == "user":
                    messages.append({
                        "role": "user", 
                        "content": data_prompt_item["content"]
                    })

            completion_length = inputs_data['completion_mask'][i].sum().item()
            completion_tokens = completion_ids[i][:completion_length]
            completion_text = self.processing_class.decode(completion_tokens, skip_special_tokens=False)

            messages += [
                {"role": "assistant", "content": completion_text},
                {"role": "user", "content": evaluate_request}
            ]

            # 评估输出答案
            clear_completion_text = completion_text
            if clear_completion_text.endswith('<|im_end|>'):
                clear_completion_text = clear_completion_text[:-len('<|im_end|>')]
            evaluation_string = self.reward_evaluator.evaluate_and_format(
                completion=clear_completion_text,
                solution=self._current_sample['solution']
            )
            evaluation_data.append({
                "messages": messages,
                "evaluation_str": evaluation_string,
            })
        
        ## 第二步，打包成 sft 数据格式
        max_padding_length = -1
        before_padding_samples = []

        for sample in evaluation_data:
            messages = sample["messages"]
            evaluation_str = sample["evaluation_str"]
            full_messages = messages + [{"role": "assistant", "content": evaluation_str}]
            
            # 处理单个样本
            full_ids, attention_mask, labels = self.prepare_single_sft_sample(messages, full_messages)
            max_padding_length = max(max_padding_length, len(full_ids))
            before_padding_samples.append({
                "input_ids": torch.tensor(full_ids),
                "attention_mask": torch.tensor(attention_mask),
                "labels": torch.tensor(labels)
            })

        batch_input_ids = []
        batch_attention_masks = []
        batch_labels = []

        for processed_sample in before_padding_samples:
            padded_sample = pad_sample_to_length(
                processed_sample, 
                max_padding_length, 
                pad_token_id
            )
            batch_input_ids.append(padded_sample["input_ids"])
            batch_attention_masks.append(padded_sample["attention_mask"])
            batch_labels.append(padded_sample["labels"])

        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_masks),
            "labels": torch.stack(batch_labels)
        }


    def calculate_mixed_loss(self, grpo_loss, stage1_sft_loss, stage2_sft_loss):
        # 计算总loss
        assert self.stage1_grpo_weight or self.stage1_sft_weight or self.stage2_sft_weight, \
            "At least one training stage must be enabled."
        total_loss = torch.tensor(0.0, device=grpo_loss.device, dtype=grpo_loss.dtype)

        if self.stage1_grpo_weight:
            total_loss = total_loss + grpo_loss * self.stage1_grpo_weight
        if self.stage1_sft_weight:
            total_loss = total_loss + stage1_sft_loss * self.stage1_sft_weight
        if self.stage2_sft_weight:
            total_loss = total_loss + stage2_sft_loss * self.stage2_sft_weight
        return total_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        计算混合损失: GRPO loss + SFT loss
        
        Args:
            model: 训练模型
            inputs: _prepare_inputs返回的输入数据
        """
        
        # 1. 计算GRPO损失 (使用父类方法)
        # [fix] 可能计算额外的 KL 散度等，带来更多的 loss 项。如果不用，就不算。
        if self.stage1_grpo_weight:
            grpo_loss = super()._compute_loss(model, inputs)
        else:
            grpo_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # 2. 第一阶段的推理过程
        if self.stage1_sft_weight:
            stage1_sft_inputs = self.prepare_stage1_sft_inputs()
            stage1_sft_inputs = {k: v.to(model.device) for k, v in stage1_sft_inputs.items()}
            
            stage1_sft_outputs = model(
                input_ids=stage1_sft_inputs["input_ids"],
                attention_mask=stage1_sft_inputs["attention_mask"],
                labels=stage1_sft_inputs["labels"]
            )
            stage1_sft_loss = stage1_sft_outputs.loss
        else:
            # 如果没有SFT数据，SFT损失为0
            stage1_sft_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # 3. 第二阶段的评估过程
        if self.stage2_sft_weight:
            stage2_sft_inputs = self.prepare_stage2_sft_inputs(inputs)
            stage2_sft_inputs = {k: v.to(model.device) for k, v in stage2_sft_inputs.items()}

            stage2_sft_outputs = model(
                input_ids=stage2_sft_inputs["input_ids"],
                attention_mask=stage2_sft_inputs["attention_mask"],
                labels=stage2_sft_inputs["labels"]
            )
            stage2_sft_loss = stage2_sft_outputs.loss
        else:
            stage2_sft_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
        
        # 4. 计算总损失
        total_loss = self.calculate_mixed_loss(grpo_loss, stage1_sft_loss, stage2_sft_loss)
        
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["grpo_loss"].append(grpo_loss.item())
        self._metrics[mode]["stage1_sft_loss"].append(stage1_sft_loss.item())
        self._metrics[mode]["stage2_sft_loss"].append(stage2_sft_loss.item())
        self._metrics[mode]["total_loss"].append(total_loss.item())
        # self._last_total_loss = total_loss.item()
        return total_loss

    # def log(self, logs: Dict[str, float]) -> None:
    #     """确保记录正确的损失值"""
    #     # 替换loss字段
    #     if hasattr(self, '_last_total_loss'):
    #         logs['grpo_original_loss'] = logs.get('loss', 0.0)  # 保存原始GRPO loss
    #         logs['loss'] = self._last_total_loss  # 使用我们的total_loss
        
    #     super().log(logs)