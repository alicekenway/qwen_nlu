"""
Supervised Fine-Tuning (SFT) for Qwen2 with PEFT LoRA

This script implements proper data processing for Alpaca format datasets.

Data format supported:
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- Supports {{name}} and {{author}} placeholders for identity training
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizer,
    Trainer,
)
from trl import SFTConfig

# Ignore index for labels (tokens that shouldn't contribute to loss)
IGNORE_INDEX = -100


# =============================================================================
# Qwen2 ChatML Template (matching LLaMA-Factory format)
# =============================================================================
@dataclass
class Qwen2Template:
    """ChatML template for Qwen2 models.
    
    Format:
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
    {assistant}<|im_end|>
    
    Note: The '<|im_start|>assistant\n' is placed at the END of user format,
    so it gets masked in labels. Only the assistant content is trained on.
    """
    # System format
    system_format: str = "<|im_start|>system\n{content}<|im_end|>\n"
    # User format includes the assistant prefix at the end (for proper label masking)
    user_format: str = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    # Assistant format is just the content + end token
    assistant_format: str = "{content}<|im_end|>\n"
    # Default system prompt
    default_system: str = "You are a helpful assistant."
    
    def encode_with_labels(
        self,
        tokenizer: PreTrainedTokenizer,
        user_content: str,
        assistant_content: str,
        system: Optional[str] = None,
    ) -> tuple[list[int], list[int]]:
        """Encode a single turn conversation and create labels.
        
        - System message: masked (not trained on)
        - User message + assistant prefix: masked (not trained on)  
        - Assistant response: trained on
        """
        input_ids = []
        labels = []
        
        # 1. Add system message (masked)
        system_content = system or self.default_system
        system_text = self.system_format.format(content=system_content)
        system_ids = tokenizer.encode(system_text, add_special_tokens=False)
        input_ids.extend(system_ids)
        labels.extend([IGNORE_INDEX] * len(system_ids))
        
        # 2. Add user message with assistant prefix (masked)
        user_text = self.user_format.format(content=user_content)
        user_ids = tokenizer.encode(user_text, add_special_tokens=False)
        input_ids.extend(user_ids)
        labels.extend([IGNORE_INDEX] * len(user_ids))
        
        # 3. Add assistant response (trained on)
        assistant_text = self.assistant_format.format(content=assistant_content)
        assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        input_ids.extend(assistant_ids)
        labels.extend(assistant_ids)  # Train on these tokens
        
        return input_ids, labels


# =============================================================================
# Alpaca Data Converter
# =============================================================================
def convert_alpaca_example(
    example: dict[str, Any],
    name: str = "Qwen",
    author: str = "Alibaba Cloud",
) -> dict[str, str]:
    """Convert Alpaca format example to user/assistant content.
    
    Handles placeholder replacement for identity training:
    - {{name}} -> model name
    - {{author}} -> author/developer name
    
    Args:
        example: Dict with 'instruction', 'input' (optional), 'output'
        name: Name to replace {{name}} placeholder
        author: Author to replace {{author}} placeholder
    
    Returns:
        Dict with 'user_content', 'assistant_content', 'system'
    """
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")
    system = example.get("system", "")
    
    # Combine instruction and input for user content
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction
    
    # Replace identity placeholders
    output = output.replace("{{name}}", name).replace("{{author}}", author)
    user_content = user_content.replace("{{name}}", name).replace("{{author}}", author)
    system = system.replace("{{name}}", name).replace("{{author}}", author)
    
    return {
        "user_content": user_content,
        "assistant_content": output,
        "system": system,
    }


# =============================================================================
# Dataset Processor
# =============================================================================
class SFTDatasetProcessor:
    """Process Alpaca format dataset for SFT training."""
    
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        template: Qwen2Template,
        max_seq_length: int = 2048,
        model_name: str = "Qwen",
        model_author: str = "Alibaba Cloud",
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.max_seq_length = max_seq_length
        self.model_name = model_name
        self.model_author = model_author
    
    def process_example(self, example: dict[str, Any]) -> dict[str, Any]:
        """Process a single example for training."""
        # Convert Alpaca format
        converted = convert_alpaca_example(
            example,
            name=self.model_name,
            author=self.model_author,
        )
        
        # Skip empty examples
        if not converted["user_content"] or not converted["assistant_content"]:
            return {
                "input_ids": [],
                "attention_mask": [],
                "labels": [],
            }
        
        # Encode with labels
        input_ids, labels = self.template.encode_with_labels(
            self.tokenizer,
            user_content=converted["user_content"],
            assistant_content=converted["assistant_content"],
            system=converted["system"] if converted["system"] else None,
        )
        
        # Truncate if necessary
        if len(input_ids) > self.max_seq_length:
            input_ids = input_ids[:self.max_seq_length]
            labels = labels[:self.max_seq_length]
        
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": labels,
        }


# =============================================================================
# Data Collator
# =============================================================================
@dataclass
class SFTDataCollator:
    """Data collator for SFT with proper padding and label masking."""
    
    tokenizer: PreTrainedTokenizer
    max_seq_length: int = 2048
    pad_to_multiple_of: int = 8
    
    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Get max length in batch
        max_len = max(len(f["input_ids"]) for f in features)
        
        # Pad to multiple of pad_to_multiple_of
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
        max_len = min(max_len, self.max_seq_length)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for feature in features:
            input_ids = feature["input_ids"][:max_len]
            labels = feature["labels"][:max_len]
            
            # Calculate padding length
            pad_len = max_len - len(input_ids)
            
            # Pad sequences
            batch_input_ids.append(input_ids + [self.tokenizer.pad_token_id] * pad_len)
            batch_attention_mask.append([1] * len(input_ids) + [0] * pad_len)
            batch_labels.append(labels + [IGNORE_INDEX] * pad_len)
        
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# =============================================================================
# Arguments
# =============================================================================
@dataclass
class ModelArguments:
    """Arguments for model configuration."""
    model_name_or_path: str = field(
        default="Qwen/Qwen2-0.5B",
        metadata={"help": "Path to pretrained model or model identifier"}
    )
    use_4bit: bool = field(
        default=False,
        metadata={"help": "Use 4-bit quantization"}
    )
    use_8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit quantization"}
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Trust remote code from HuggingFace"}
    )
    use_flash_attention: bool = field(
        default=True,
        metadata={"help": "Use flash attention if available"}
    )


@dataclass
class LoraArguments:
    """Arguments for LoRA configuration."""
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target modules for LoRA"}
    )


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to local JSON/JSONL dataset file"}
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "Dataset name from HuggingFace Hub"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of preprocessing workers"}
    )
    # Identity placeholders
    model_name: str = field(
        default="Qwen",
        metadata={"help": "Model name for {{name}} placeholder replacement"}
    )
    model_author: str = field(
        default="Alibaba Cloud",
        metadata={"help": "Author for {{author}} placeholder replacement"}
    )


# =============================================================================
# Helper Functions
# =============================================================================
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
        return BitsAndBytesConfig(load_in_8bit=True)
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


def load_local_dataset(path: str) -> Dataset:
    """Load dataset from local JSON/JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        if path.endswith('.jsonl'):
            data = [json.loads(line) for line in f if line.strip()]
        else:
            data = json.load(f)
            if not isinstance(data, list):
                data = [data]
    return Dataset.from_list(data)


def load_model_and_tokenizer(model_args: ModelArguments, lora_args: LoraArguments):
    """Load model and tokenizer with LoRA configuration."""
    
    bnb_config = create_bnb_config(model_args.use_4bit, model_args.use_8bit)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Determine attention implementation
    attn_impl = None
    if model_args.use_flash_attention and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("Flash attention not available, using default attention")
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation=attn_impl,
    )
    
    model.gradient_checkpointing_enable()
    
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    lora_config = create_lora_config(lora_args)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# =============================================================================
# Main
# =============================================================================
def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, SFTConfig))
    model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(model_args, lora_args)
    
    # Create template and processor
    template = Qwen2Template()
    processor = SFTDatasetProcessor(
        tokenizer=tokenizer,
        template=template,
        max_seq_length=data_args.max_seq_length,
        model_name=data_args.model_name,
        model_author=data_args.model_author,
    )
    
    # Load dataset
    print("Loading dataset...")
    if data_args.dataset_path:
        dataset = load_local_dataset(data_args.dataset_path)
        print(f"Loaded {len(dataset)} examples from {data_args.dataset_path}")
    elif data_args.dataset_name:
        dataset = load_dataset(data_args.dataset_name, split="train")
        print(f"Loaded {len(dataset)} examples from {data_args.dataset_name}")
    else:
        raise ValueError("Either dataset_path or dataset_name must be specified")
    
    # Print sample before processing
    print("\n=== Sample raw data ===")
    sample = dataset[0]
    for key, value in sample.items():
        print(f"{key}: {str(value)[:100]}...")
    
    # Process dataset
    print("\nProcessing dataset...")
    processed_dataset = dataset.map(
        processor.process_example,
        remove_columns=dataset.column_names,
        num_proc=data_args.preprocessing_num_workers,
        desc="Processing dataset",
    )
    
    # Filter empty examples
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering empty examples",
    )
    print(f"Processed {len(processed_dataset)} examples")
    
    # Print sample after processing
    print("\n=== Sample processed data ===")
    sample = processed_dataset[0]
    print(f"input_ids length: {len(sample['input_ids'])}")
    print(f"labels length: {len(sample['labels'])}")
    
    # Decode and show
    valid_labels = [l for l in sample['labels'] if l != IGNORE_INDEX]
    print(f"\nFull input:\n{tokenizer.decode(sample['input_ids'], skip_special_tokens=False)}")
    print(f"\nLabels (what model learns):\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")
    
    # Create data collator and trainer
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n=== Starting training ===")
    trainer.train()
    
    # Save
    print(f"\nSaving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\n=== Training complete! ===")


if __name__ == "__main__":
    main()
