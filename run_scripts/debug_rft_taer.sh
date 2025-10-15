accelerate launch --main_process_port=23498 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/rft_taer.py \
    --config recipes/Qwen2.5-7B-Instruct/rftv2/debug_taer.yaml \
    --vllm_mode "colocate"
