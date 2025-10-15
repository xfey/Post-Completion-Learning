accelerate launch --main_process_port=23498 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-7B-Instruct/grpo/config_math220k.yaml \
    --vllm_mode "colocate"
