accelerate launch --main_process_port=23498 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/rft.py \
    --config recipes/Qwen2.5-7B-Instruct/rft/debug.yaml \
    --vllm_mode "colocate"
