#!/bin/bash
# =============================================================================
# Qwen2 SFT Training Launch Script
# =============================================================================
# Usage:
#   Single GPU:  bash run_train.sh
#   Multi-GPU:   bash run_train.sh --num_gpus 4
#   With config: bash run_train.sh --config my_config.yaml
#   DeepSpeed:   bash run_train.sh --deepspeed ds_config.json
# =============================================================================

set -e  # Exit on error

# -----------------------------------------------------------------------------
# Default Configuration
# -----------------------------------------------------------------------------
NUM_GPUS=1
CONFIG_FILE="config_example.yaml"
DEEPSPEED_CONFIG=""
MASTER_PORT=29500

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -----------------------------------------------------------------------------
# Parse Arguments
# -----------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --deepspeed)
            DEEPSPEED_CONFIG="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            # Pass remaining args to training script
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# CUDA settings
export CUDA_DEVICE_MAX_CONNECTIONS=1

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# -----------------------------------------------------------------------------
# Print Configuration
# -----------------------------------------------------------------------------
echo "=============================================="
echo "Qwen2 SFT Training"
echo "=============================================="
echo "Script directory: $SCRIPT_DIR"
echo "Config file: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "Master port: $MASTER_PORT"
if [ -n "$DEEPSPEED_CONFIG" ]; then
    echo "DeepSpeed config: $DEEPSPEED_CONFIG"
fi
if [ -n "$EXTRA_ARGS" ]; then
    echo "Extra arguments: $EXTRA_ARGS"
fi
echo "=============================================="

# -----------------------------------------------------------------------------
# Check Config File Exists
# -----------------------------------------------------------------------------
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Launch Training
# -----------------------------------------------------------------------------
if [ "$NUM_GPUS" -eq 1 ]; then
    # Single GPU training
    echo "Starting single GPU training..."
    
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        python train.py "$CONFIG_FILE" \
            --deepspeed "$DEEPSPEED_CONFIG" \
            $EXTRA_ARGS
    else
        python train.py "$CONFIG_FILE" $EXTRA_ARGS
    fi

else
    # Multi-GPU training with accelerate
    echo "Starting multi-GPU training with $NUM_GPUS GPUs..."
    
    if [ -n "$DEEPSPEED_CONFIG" ]; then
        # With DeepSpeed
        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $MASTER_PORT \
            --use_deepspeed \
            --deepspeed_config_file "$DEEPSPEED_CONFIG" \
            train.py "$CONFIG_FILE" $EXTRA_ARGS
    else
        # Standard DDP
        accelerate launch \
            --num_processes $NUM_GPUS \
            --main_process_port $MASTER_PORT \
            --multi_gpu \
            train.py "$CONFIG_FILE" $EXTRA_ARGS
    fi
fi

echo ""
echo "=============================================="
echo "Training completed!"
echo "=============================================="

