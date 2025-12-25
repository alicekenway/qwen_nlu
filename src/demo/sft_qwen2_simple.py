"""
Simple SFT script for Qwen2-0.5B with PEFT LoRA

A minimal, ready-to-run example for fine-tuning Qwen2-0.5B.
Just modify the dataset and run!
"""

import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig


# =============================================================================
# Configuration - Modify these as needed
# =============================================================================

MODEL_NAME = "Qwen/Qwen2-0.5B"  # or "Qwen/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./output/qwen2-sft-lora"
MAX_SEQ_LENGTH = 512

# LoRA Configuration
LORA_R = 16          # LoRA rank
LORA_ALPHA = 32      # LoRA alpha (scaling factor)
LORA_DROPOUT = 0.05  # LoRA dropout

# Training Configuration
EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4


# =============================================================================
# Custom Dataset - Modify this section for your data
# =============================================================================

def create_sample_dataset() -> Dataset:
    """
    Create a sample dataset for demonstration.
    Replace this with your own data loading logic.
    """
    # Example: Simple instruction-following data
    data = [
        {
            "instruction": "What is the capital of France?",
            "response": "The capital of France is Paris."
        },
        {
            "instruction": "Translate 'Hello' to Spanish.",
            "response": "Hello in Spanish is 'Hola'."
        },
        {
            "instruction": "Write a haiku about programming.",
            "response": "Code flows like water\nBugs hide in the deepest lines\nDebug brings the light"
        },
        {
            "instruction": "Explain what machine learning is in simple terms.",
            "response": "Machine learning is a type of artificial intelligence where computers learn patterns from data instead of being explicitly programmed. It's like teaching a computer to recognize cats by showing it thousands of cat pictures, rather than writing rules about what a cat looks like."
        },
        {
            "instruction": "What are the benefits of exercise?",
            "response": "Exercise has many benefits including improved cardiovascular health, stronger muscles and bones, better mental health, increased energy levels, better sleep quality, and reduced risk of chronic diseases like diabetes and heart disease."
        },
    ]
    
    return Dataset.from_list(data)


def format_chat(sample: dict) -> str:
    """
    Format sample into Qwen2 chat format.
    Customize based on your data structure.
    """
    instruction = sample.get("instruction", "")
    response = sample.get("response", "")
    
    # Qwen2 chat format
    return f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{response}<|im_end|>"


# =============================================================================
# Main Training Logic
# =============================================================================

def main():
    print(f"🚀 Starting SFT for {MODEL_NAME}")
    
    # Load tokenizer
    print("📦 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    print("📦 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Configure LoRA
    print("🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
            "gate_proj", "up_proj", "down_proj",      # MLP layers
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    print("📊 Loading dataset...")
    # Option 1: Use sample dataset
    dataset = create_sample_dataset()
    
    # Option 2: Load from HuggingFace Hub (uncomment to use)
    # dataset = load_dataset("timdettmers/openassistant-guanaco", split="train")
    
    # Option 3: Load from local JSON/CSV file (uncomment to use)
    # dataset = load_dataset("json", data_files="your_data.json", split="train")
    
    print(f"   Dataset size: {len(dataset)}")
    
    # Configure training
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        max_seq_length=MAX_SEQ_LENGTH,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        optim="adamw_torch",
        report_to="none",
        remove_unused_columns=False,
    )
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=format_chat,
    )
    
    # Train!
    print("🏋️ Starting training...")
    trainer.train()
    
    # Save
    print(f"💾 Saving model to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("✅ Training complete!")
    print(f"   Model saved to: {OUTPUT_DIR}")
    print("\n📝 To use the trained model:")
    print(f"""
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("{MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}")

# Load LoRA weights
model = PeftModel.from_pretrained(base_model, "{OUTPUT_DIR}")

# Or merge LoRA into base model for faster inference
model = model.merge_and_unload()
""")


if __name__ == "__main__":
    main()

