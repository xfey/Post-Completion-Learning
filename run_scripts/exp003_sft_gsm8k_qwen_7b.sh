### Run SFT
accelerate launch --main_process_port=23498 --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-7B-Instruct/sft/exp003_config_gsm8k.yaml
