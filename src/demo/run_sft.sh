#!/bin/bash
# SFT training for Qwen2 with LoRA - Alpaca format data

set -e

export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY_DATA="${SCRIPT_DIR}/../../LLaMA-Factory/data"

# =============================================================================
# Example 1: Train with alpaca_en_demo.json
# =============================================================================
run_alpaca() {
    echo "=== Training with Alpaca English demo data ==="
    python "${SCRIPT_DIR}/sft_qwen2_lora.py" \
        --model_name_or_path "Qwen/Qwen2-0.5B" \
        --dataset_path "${LLAMA_FACTORY_DATA}/alpaca_en_demo.json" \
        --output_dir "${SCRIPT_DIR}/output/qwen2-alpaca-lora" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-4 \
        --max_seq_length 1024 \
        --lora_r 16 \
        --lora_alpha 32 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 3 \
        --bf16 \
        --gradient_checkpointing \
        --optim "adamw_torch" \
        --report_to "none"
}

# =============================================================================
# Example 2: Train with identity.json (custom model identity)
# =============================================================================
run_identity() {
    echo "=== Training with identity data (custom name/author) ==="
    python "${SCRIPT_DIR}/sft_qwen2_lora.py" \
        --model_name_or_path "Qwen/Qwen2-0.5B" \
        --dataset_path "${LLAMA_FACTORY_DATA}/identity.json" \
        --model_name "MyAssistant" \
        --model_author "MyCompany" \
        --output_dir "${SCRIPT_DIR}/output/qwen2-identity-lora" \
        --num_train_epochs 5 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --max_seq_length 512 \
        --lora_r 16 \
        --lora_alpha 32 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --save_steps 50 \
        --save_total_limit 3 \
        --bf16 \
        --gradient_checkpointing \
        --optim "adamw_torch" \
        --report_to "none"
}

# =============================================================================
# Example 3: Train with 4-bit quantization (lower memory)
# =============================================================================
run_4bit() {
    echo "=== Training with 4-bit quantization ==="
    python "${SCRIPT_DIR}/sft_qwen2_lora.py" \
        --model_name_or_path "Qwen/Qwen2-0.5B" \
        --dataset_path "${LLAMA_FACTORY_DATA}/alpaca_en_demo.json" \
        --use_4bit \
        --output_dir "${SCRIPT_DIR}/output/qwen2-4bit-lora" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 2 \
        --learning_rate 2e-4 \
        --max_seq_length 1024 \
        --lora_r 16 \
        --lora_alpha 32 \
        --warmup_ratio 0.03 \
        --lr_scheduler_type "cosine" \
        --logging_steps 10 \
        --save_steps 100 \
        --save_total_limit 3 \
        --bf16 \
        --gradient_checkpointing \
        --optim "adamw_torch" \
        --report_to "none"
}

# =============================================================================
# Main
# =============================================================================
case "${1:-alpaca}" in
    alpaca)
        run_alpaca
        ;;
    identity)
        run_identity
        ;;
    4bit)
        run_4bit
        ;;
    *)
        echo "Usage: $0 {alpaca|identity|4bit}"
        echo ""
        echo "Examples:"
        echo "  $0 alpaca   - Train with alpaca_en_demo.json"
        echo "  $0 identity - Train with identity.json (custom model name/author)"
        echo "  $0 4bit     - Train with 4-bit quantization"
        exit 1
        ;;
esac
