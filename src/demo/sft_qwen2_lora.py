"""
Supervised Fine-Tuning (SFT) for Qwen2-0.5B with PEFT LoRA

This script demonstrates how to fine-tune Qwen2-0.5B using LoRA (Low-Rank Adaptation)
for efficient parameter-efficient fine-tuning.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig


@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-0.5B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 4-bit quantization for memory efficiency"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Whether to use 8-bit quantization for memory efficiency"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code from HuggingFace"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(
        default=16,
        metadata={"help": "LoRA attention dimension (rank)"}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter for scaling"}
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={"help": "LoRA dropout probability"}
    )
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target modules for LoRA"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_name: str = field(
        default="timdettmers/openassistant-guanaco",
        metadata={"help": "Dataset name or path"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length for training"}
    )
    dataset_text_field: str = field(
        default="text",
        metadata={"help": "Field name containing the text data"}
    )


def create_bnb_config(use_4bit: bool = False, use_8bit: bool = False) -> Optional[BitsAndBytesConfig]:
    """Create BitsAndBytes quantization config."""
    if use_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif use_8bit:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def create_lora_config(lora_args: LoraArguments) -> LoraConfig:
    """Create LoRA configuration."""
    target_modules = lora_args.target_modules.split(",") if lora_args.target_modules else None
    
    return LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def format_instruction(sample: dict) -> str:
    """Format a sample into instruction format for Qwen2."""
    # Customize this based on your dataset format
    if "instruction" in sample and "response" in sample:
        instruction = sample["instruction"]
        response = sample["response"]
        context = sample.get("context", "")
        
        if context:
            return f"<|im_start|>user\n{instruction}\n\nContext: {context}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
        return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"
    
    elif "text" in sample:
        return sample["text"]
    
    elif "messages" in sample:
        # Handle chat format
        formatted = ""
        for msg in sample["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        return formatted.strip()
    
    return str(sample)


def load_model_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """Load model and tokenizer with LoRA configuration."""
    
    # Create quantization config if needed
    bnb_config = create_bnb_config(model_args.use_4bit, model_args.use_8bit)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    
    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Prepare model for k-bit training if using quantization
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Create LoRA config and apply to model
    lora_config = create_lora_config(lora_args)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model, tokenizer


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, SFTConfig))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args)
    
    # Load dataset
    print(f"Loading dataset: {data_args.dataset_name}")
    dataset = load_dataset(data_args.dataset_name, split="train")
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=format_instruction,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("Training complete!")


if __name__ == "__main__":
    main()

