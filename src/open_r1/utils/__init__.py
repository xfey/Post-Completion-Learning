from .data import get_dataset
from .import_utils import is_e2b_available, is_morph_available
from .model_utils import get_model, get_tokenizer


__all__ = ["get_tokenizer", "is_e2b_available", "is_morph_available", "get_model", "get_dataset", 
           "get_rft_loss_args"]


def get_rft_loss_args(script_args):
    stage1_grpo_weight = script_args.stage1_grpo_weight
    stage1_sft_weight = script_args.stage1_sft_weight
    stage2_sft_weight = script_args.stage2_sft_weight
    return {
        "stage1_grpo_weight": stage1_grpo_weight,
        "stage1_sft_weight": stage1_sft_weight,
        "stage2_sft_weight": stage2_sft_weight,
    }
