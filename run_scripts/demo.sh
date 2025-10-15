# # Train via command line
# accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
#     --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
#     --dataset_name open-r1/Mixture-of-Thoughts \
#     --dataset_config all \
#     --eos_token '<|im_end|>' \
#     --learning_rate 4.0e-5 \
#     --num_train_epochs 5 \
#     --max_seq_length 32768 \
#     --per_device_train_batch_size 2 \
#     --gradient_checkpointing \
#     --bf16 \
#     --use_liger_kernel \
#     --output_dir data/OpenR1-Distill-7B

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/OpenR1-Distill-7B/sft/config_distill.yaml