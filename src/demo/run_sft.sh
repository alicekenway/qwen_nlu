#!/bin/bash
# Example script to run SFT training for Qwen2-0.5B with LoRA

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true  # Disable wandb if not using

python sft_qwen2_lora.py \
    --model_name_or_path "Qwen/Qwen2-0.5B" \
    --dataset_name "timdettmers/openassistant-guanaco" \
    --output_dir "./output/qwen2-0.5b-sft-lora" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --max_seq_length 512 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_steps 100 \
    --save_total_limit 3 \
    --bf16 \
    --gradient_checkpointing \
    --optim "adamw_torch" \
    --report_to "none"

