import re
from typing import List, Dict, Any, Optional, Callable
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify


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
                eval_str = self._format_accuracy_evaluation(detail, reward)
            elif func_name == "format_reward":
                reward, detail = self._evaluate_format_detailed(completion, **kwargs)
                eval_str = self._format_format_evaluation(detail, reward)
            elif func_name == "tag_count_reward":
                reward, detail = self._evaluate_tag_count_detailed(completion, **kwargs)
                eval_str = self._format_tag_count_evaluation(detail, reward)
            else:
                # 对于其他奖励函数，使用默认方式
                # 需要转换为原始函数期望的格式
                completions_batch = [[{"content": completion}]]
                solutions_batch = [solution]
                rewards = func(completions_batch, solutions=solutions_batch, **kwargs)
                reward = rewards[0] if rewards else None
                eval_str = f"{func_name}: reward={reward}"
            
            eval_parts.append(eval_str)
        
        # 合并所有评估部分
        full_evaluation = "\n".join(eval_parts)
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
            "gold_parsing": "unknown",
            "answer_extraction": "unknown",
            "extracted_answer": None,
            "gold_answer": None,
            "verification": "unknown",
            "error_message": None
        }
        
        try:
            # 解析正确答案
            gold_parsed = parse(solution, extraction_mode="first_match")
            
            if len(gold_parsed) != 0:
                detail["gold_parsing"] = "success"
                detail["gold_answer"] = str(gold_parsed)
                
                # 解析模型答案
                answer_parsed = parse(completion, extraction_mode="first_match")
                if not answer_parsed:
                    answer_parsed = parse(
                        completion,
                        extraction_config=[
                            LatexExtractionConfig(
                                normalization_config=NormalizationConfig(
                                    nits=False,
                                    malformed_operators=False,
                                    basic_latex=True,
                                    equations=True,
                                    boxed="all",
                                    units=True,
                                ),
                                boxed_match_priority=0,
                                try_extract_without_anchor=False,
                            )
                        ],
                        extraction_mode="first_match",
                    )
                
                if len(answer_parsed) > 0:
                    detail["answer_extraction"] = "success"
                    detail["extracted_answer"] = str(answer_parsed)
                    
                    # 验证答案
                    try:
                        reward = float(verify(gold_parsed, answer_parsed))
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
                detail["gold_parsing"] = "failed"
                detail["verification"] = "skipped"
                reward = None
                
        except Exception as e:
            detail["gold_parsing"] = "error"
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
    
    def _format_accuracy_evaluation(self, detail: Dict[str, Any], reward: Optional[float]) -> str:
        """格式化准确性评估字符串"""
        if reward is None:
            if detail["gold_parsing"] == "failed":
                return f"accuracy: gold_parsing=failed, reason=\"unparseable_solution\" → reward=null"
            elif detail["answer_extraction"] == "failed":
                return f"accuracy: answer_extraction=failed, reason=\"no_answer_found\" → reward=null"
            elif detail["verification"] == "error":
                return f"accuracy: verification=error, reason=\"{detail.get('error_message', 'unknown')}\" → reward=null"
            else:
                return f"accuracy: evaluation=failed, reason=\"unknown_error\" → reward=null"
        else:
            extracted = detail.get("extracted_answer", "unknown")
            gold = detail.get("gold_answer", "unknown")
            verification = detail["verification"]
            return f"accuracy: extracted_answer=\"{extracted}\", gold_answer=\"{gold}\", verification={verification} → reward={reward}"
    
    def _format_format_evaluation(self, detail: Dict[str, Any], reward: float) -> str:
        """格式化格式评估字符串"""
        has_think = "yes" if detail["has_think_tags"] else "no"
        has_answer = "yes" if detail["has_answer_tags"] else "no"
        structure_valid = "yes" if detail["structure_valid"] else "no"
        
        return f"format: has_think_tags={has_think}, has_answer_tags={has_answer}, structure_valid={structure_valid} → reward={reward}"
    
    def _format_tag_count_evaluation(self, detail: Dict[str, Any], reward: float) -> str:
        """格式化标签计数评估字符串"""
        think_open = detail["think_open_count"]
        think_close = detail["think_close_count"]
        answer_open = detail["answer_open_count"]
        answer_close = detail["answer_close_count"]
        
        return f"tag_count: think_open={think_open}/1, think_close={think_close}/1, answer_open={answer_open}/1, answer_close={answer_close}/1 → reward={reward}"