import re
import random
from typing import List, Dict, Any, Optional, Callable
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify



def parse_gsm8k_answer(text, mode="first_match" ):
    pattern = r"<answer>(.*?)</answer>"
    results = re.findall(pattern, text, re.DOTALL)

    if isinstance(results, list) and len(results) > 0:
        if mode == "first_match":
            results = results[0]
        else:
            results = results[-1]  
    else:
        results = ""
    return results.strip()
    

def verify_gsm8k(answer_parsed, gold_parsed):
    if answer_parsed.strip() == gold_parsed.strip():
        return True
    return False


class RewardEvaluator:
    """评估器类，用于生成详细的奖励评估字符串"""
    
    def __init__(self, reward_funcs: List[Callable]):
        """
        初始化评估器
        
        Args:
            reward_funcs: 奖励函数列表
        """
        self.reward_funcs = reward_funcs
        self.reward_func_names = [func.__name__ for func in reward_funcs]
        
        # 定义多样化的评估模板
        self.accuracy_templates = [
            # # 模板1：简洁对比风格
            # lambda d, r: f"accuracy: {d.get('extracted_answer', 'None')} vs {d.get('gold_answer', 'None')} → {self._get_match_status(d, r)} → reward={r}",
            
            # 模板2：详细分析风格
            lambda d, r: f"accuracy evaluation:\n  - Model output: {d.get('extracted_answer', 'not extracted')}\n  - Expected: {d.get('gold_answer', 'unknown')}\n  - Match status: {self._get_match_status(d, r)}\n  → reward={r}",
            
            # # 模板3：步骤追踪风格
            # lambda d, r: f"accuracy check: parsing={d['gold_parsing']}, extraction={d['answer_extraction']}, verification={d['verification']} → reward={r}",
            
            # # 模板4：自然语言风格
            # lambda d, r: f"accuracy: The model {'correctly' if r == 1.0 else 'incorrectly'} answered with \"{d.get('extracted_answer', 'no answer')}\" (expected: \"{d.get('gold_answer', 'unknown')}\") → reward={r}",
            
            # # 模板5：数学符号风格
            # lambda d, r: f"accuracy: answer={d.get('extracted_answer', '∅')}, gold={d.get('gold_answer', '?')}, answer{'=' if r == 1.0 else '≠'}gold → reward={r}"
        ]
        
        self.format_templates = [
            # 模板1：布尔风格
            lambda d, r: f"format: think={d['has_think_tags']}, answer={d['has_answer_tags']}, valid={d['structure_valid']} → reward={r}",
            
            # # 模板2：检查清单风格
            # lambda d, r: f"format check:\n  ✓ <think> tags: {'yes' if d['has_think_tags'] else 'no'}\n  ✓ <answer> tags: {'yes' if d['has_answer_tags'] else 'no'}\n  ✓ Structure: {'valid' if d['structure_valid'] else 'invalid'}\n  → reward={r}",
            
            # # 模板3：状态码风格
            # lambda d, r: f"format: tags_status={'OK' if d['has_think_tags'] and d['has_answer_tags'] else 'MISSING'}, structure={'VALID' if d['structure_valid'] else 'INVALID'} → reward={r}",
            
            # # 模板4：简化风格
            # lambda d, r: f"format: {'all tags present and valid' if r == 1.0 else 'missing tags or invalid structure'} → reward={r}",
            
            # # 模板5：符号风格
            # lambda d, r: f"format: think={'✓' if d['has_think_tags'] else '✗'}, answer={'✓' if d['has_answer_tags'] else '✗'}, structure={'✓' if d['structure_valid'] else '✗'} → reward={r}"
        ]
        
        self.tag_count_templates = [
            # 模板1：计数风格
            lambda d, r: f"tag_count: <think>={d['think_open_count']}, </think>={d['think_close_count']}, <answer>={d['answer_open_count']}, </answer>={d['answer_close_count']} → reward={r}",
            
            # # 模板2：比例风格
            # lambda d, r: f"tag_count: found {int(r*4)}/4 required tags → reward={r}",
            
            # # 模板3：详细分解风格
            # lambda d, r: f"tag_count scoring:\n  - think_open: {d['score_breakdown']['think_open']}\n  - think_close: {d['score_breakdown']['think_close']}\n  - answer_open: {d['score_breakdown']['answer_open']}\n  - answer_close: {d['score_breakdown']['answer_close']}\n  → total_reward={r}",
            
            # # 模板4：百分比风格
            # lambda d, r: f"tag_count: {int(r*100)}% tags correct (think: {d['think_open_count']+d['think_close_count']}/2, answer: {d['answer_open_count']+d['answer_close_count']}/2) → reward={r}",
            
            # # 模板5：缺失报告风格
            # lambda d, r: self._format_missing_tags(d, r)
        ]
        
    def _get_match_status(self, detail: Dict[str, Any], reward: Optional[float]) -> str:
        """获取匹配状态描述"""
        if reward is None:
            return "evaluation_failed"
        elif reward == 1.0:
            return "match"
        else:
            return "mismatch"
    
    def _format_missing_tags(self, detail: Dict[str, Any], reward: float) -> str:
        """格式化缺失标签报告"""
        missing = []
        if detail['think_open_count'] != 1:
            missing.append("<think>")
        if detail['think_close_count'] != 1:
            missing.append("</think>")
        if detail['answer_open_count'] != 1:
            missing.append("<answer>")
        if detail['answer_close_count'] != 1:
            missing.append("</answer>")
        
        if not missing:
            return f"tag_count: all tags present → reward={reward}"
        else:
            return f"tag_count: missing tags {', '.join(missing)} → reward={reward}"
    
    def evaluate_only(
        self,
        completion,
        solution
    ):
        reward_parts = []

        for func in self.reward_funcs:
            func_name = func.__name__

            if func_name == "accuracy_reward":
                reward, detail = self._evaluate_accuracy_detailed(completion, solution, **kwargs)
            elif func_name == "format_reward":
                reward, detail = self._evaluate_format_detailed(completion, **kwargs)
            elif func_name == "tag_count_reward":
                reward, detail = self._evaluate_tag_count_detailed(completion, **kwargs)
            reward_parts.append(f"{func_name.replace('_reward', '')} = {reward}")
        
        return ', '.join(reward_parts)

    def evaluate_and_format(
        self, 
        completion: str, 
        solution: str, 
        **kwargs
    ) -> str:
        """
        对单个completion进行评估并生成格式化的评估字符串
        
        Args:
            completion: 模型的completion结果（字符串）
            solution: 正确答案（字符串）
            **kwargs: 传递给奖励函数的额外参数
            
        Returns:
            str: 格式化的评估字符串
        """
        eval_parts = []
        
        for func in self.reward_funcs:
            func_name = func.__name__
            
            if func_name == "accuracy_reward":
                reward, detail = self._evaluate_accuracy_detailed(completion, solution, **kwargs)
                # 随机选择一个模板
                template = random.choice(self.accuracy_templates)
                eval_str = template(detail, reward)
            elif func_name == "format_reward":
                reward, detail = self._evaluate_format_detailed(completion, **kwargs)
                # 随机选择一个模板
                template = random.choice(self.format_templates)
                eval_str = template(detail, reward)
            elif func_name == "tag_count_reward":
                reward, detail = self._evaluate_tag_count_detailed(completion, **kwargs)
                # 随机选择一个模板
                template = random.choice(self.tag_count_templates)
                eval_str = template(detail, reward)
            else:
                # 对于其他奖励函数，使用默认方式
                completions_batch = [[{"content": completion}]]
                solutions_batch = [solution]
                rewards = func(completions_batch, solutions=solutions_batch, **kwargs)
                reward = rewards[0] if rewards else None
                eval_str = f"{func_name}: reward={reward}"
            
            eval_parts.append(eval_str)
        
        # 合并所有评估部分，使用随机选择的分隔符
        separators = ["\n", "\n---\n", "\n\n", " | ", "\n• "]
        separator = random.choice(separators)
        full_evaluation = separator.join(eval_parts)
        
        # 检查是否所有reward都为1.0，如果是，添加总结性评价
        all_perfect = self._check_all_perfect(eval_parts)
        if all_perfect:
            summary = self._generate_perfect_summary()
            full_evaluation += f"\n{summary}"
        
        return full_evaluation
    
    def _evaluate_accuracy_detailed(
        self, 
        completion: str, 
        solution: str, 
        **kwargs
    ) -> tuple[Optional[float], Dict[str, Any]]:
        """详细评估准确性，返回奖励和详细信息"""
        detail = {
            "gold_solution": solution,
            "completion": completion,
            # "gold_parsing": "unknown",
            "answer_extraction": "unknown",
            "extracted_answer": None,
            "gold_answer": None,
            "verification": "unknown",
            "error_message": None
        }
        
        try:
            # 解析正确答案
            # gold_parsed = parse(solution, extraction_mode="first_match")
            gold_parsed = parse_gsm8k_answer(
                sol,
                # extraction_mode="first_match",
            )
            
            if len(gold_parsed) != 0:
                # detail["gold_parsing"] = "success"
                detail["gold_answer"] = str(gold_parsed)
                
                # 解析模型答案
                # answer_parsed = parse(completion, extraction_mode="first_match")
                answer_parsed = parse_gsm8k_answer(
                    completion,
                )
                if len(answer_parsed) > 0:
                    detail["answer_extraction"] = "success"
                    detail["extracted_answer"] = str(answer_parsed)
                    
                    # 验证答案
                    try:
                        # reward = float(verify(gold_parsed, answer_parsed))
                        reward = float(verify_gsm8k(answer_parsed, gold_parsed))
                        detail["verification"] = "passed" if reward == 1.0 else "failed"
                    except Exception as e:
                        detail["verification"] = "error"
                        detail["error_message"] = str(e)
                        reward = None
                else:
                    detail["answer_extraction"] = "failed"
                    detail["verification"] = "skipped"
                    reward = None
            else:
                # detail["gold_parsing"] = "failed"
                detail["verification"] = "skipped"
                reward = None
                
        except Exception as e:
            # detail["gold_parsing"] = "error"
            detail["error_message"] = str(e)
            reward = None
        
        return reward, detail
    
    def _evaluate_format_detailed(
        self, 
        completion: str, 
        **kwargs
    ) -> tuple[float, Dict[str, Any]]:
        """详细评估格式，返回奖励和详细信息"""
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        
        detail = {
            "content": completion,
            "pattern": pattern,
            "has_think_tags": False,
            "has_answer_tags": False,
            "structure_valid": False,
            "match_result": False
        }
        
        # 检查是否有think标签
        detail["has_think_tags"] = "<think>" in completion and "</think>" in completion
        
        # 检查是否有answer标签
        detail["has_answer_tags"] = "<answer>" in completion and "</answer>" in completion
        
        # 检查完整结构
        match = re.match(pattern, completion, re.DOTALL | re.MULTILINE)
        detail["match_result"] = match is not None
        detail["structure_valid"] = detail["match_result"]
        
        reward = 1.0 if match else 0.0
        
        return reward, detail
    
    def _evaluate_tag_count_detailed(
        self, 
        completion: str, 
        **kwargs
    ) -> tuple[float, Dict[str, Any]]:
        """详细评估标签计数，返回奖励和详细信息"""
        detail = {
            "content": completion,
            "think_open_count": completion.count("<think>\n"),
            "think_close_count": completion.count("\n</think>\n"),
            "answer_open_count": completion.count("\n<answer>\n"),
            "answer_close_count": completion.count("\n</answer>"),
            "expected_each": 1,
            "score_breakdown": {}
        }
        
        # 计算每个标签的得分
        detail["score_breakdown"]["think_open"] = 0.25 if detail["think_open_count"] == 1 else 0.0
        detail["score_breakdown"]["think_close"] = 0.25 if detail["think_close_count"] == 1 else 0.0
        detail["score_breakdown"]["answer_open"] = 0.25 if detail["answer_open_count"] == 1 else 0.0
        detail["score_breakdown"]["answer_close"] = 0.25 if detail["answer_close_count"] == 1 else 0.0
        
        score = sum(detail["score_breakdown"].values())
        
        return score, detail
    
    def _check_all_perfect(self, eval_parts: List[str]) -> bool:
        """
        检查是否所有的reward都是1.0
        
        Args:
            eval_parts: 评估结果字符串列表
            
        Returns:
            bool: 是否所有reward都是1.0
        """
        for part in eval_parts:
            # 查找reward值
            if "reward=" in part:
                # 提取reward值
                reward_match = re.search(r'reward=([0-9.]+|null)', part)
                if reward_match:
                    reward_str = reward_match.group(1)
                    if reward_str == "null" or float(reward_str) != 1.0:
                        return False
            else:
                # 如果没找到reward，保守起见返回False
                return False
        return True
    
    def _generate_perfect_summary(self) -> str:
        """
        生成完美答案的总结性评价
        
        Returns:
            str: 总结性评价
        """
        summaries = [
            # "Perfect! All evaluation criteria are fully satisfied.",
            # "Excellent work - the answer meets all requirements.",
            # "The response is completely correct in all aspects.",
            # "All checks passed successfully.",
            "All rewards are 1.0"
        ]
        
        return random.choice(summaries)