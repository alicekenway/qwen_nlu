"""
Dataset processing module for Qwen2 SFT training.

Handles Alpaca format datasets with proper tokenization and label masking.

Data format supported:
- Alpaca: {"instruction": "...", "input": "...", "output": "..."}
- Supports {{name}} and {{author}} placeholders for identity training
"""

import json
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

# Ignore index for labels (tokens that shouldn't contribute to loss)
IGNORE_INDEX = -100


# =============================================================================
# Qwen2 ChatML Template
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
    system_format: str = "<|im_start|>system\n{content}<|im_end|>\n"
    user_format: str = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"
    assistant_format: str = "{content}<|im_end|>\n"
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
    
    def format_prompt(
        self,
        user_content: str,
        system: Optional[str] = None,
    ) -> str:
        """Format a prompt for inference (without assistant response)."""
        system_content = system or self.default_system
        prompt = self.system_format.format(content=system_content)
        prompt += self.user_format.format(content=user_content)
        return prompt


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
# Data Arguments
# =============================================================================
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
    eval_dataset_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation dataset (JSON/JSONL). If not set, uses val_size split from training data"}
    )
    val_size: float = field(
        default=0.0,
        metadata={"help": "Fraction of training data to use for validation (0.0-1.0). Only used if eval_dataset_path is not set"}
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessing_num_workers: int = field(
        default=4,
        metadata={"help": "Number of preprocessing workers"}
    )
    model_name: str = field(
        default="Qwen",
        metadata={"help": "Model name for {{name}} placeholder replacement"}
    )
    model_author: str = field(
        default="Alibaba Cloud",
        metadata={"help": "Author for {{author}} placeholder replacement"}
    )


# =============================================================================
# Dataset Loading Functions
# =============================================================================
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


def load_sft_dataset(data_args: DataArguments) -> Dataset:
    """Load dataset from either local file or HuggingFace Hub."""
    if data_args.dataset_path:
        dataset = load_local_dataset(data_args.dataset_path)
        print(f"Loaded {len(dataset)} examples from {data_args.dataset_path}")
    elif data_args.dataset_name:
        dataset = load_dataset(data_args.dataset_name, split="train")
        print(f"Loaded {len(dataset)} examples from {data_args.dataset_name}")
    else:
        raise ValueError("Either dataset_path or dataset_name must be specified")
    return dataset


def load_train_eval_datasets(
    data_args: DataArguments,
) -> tuple[Dataset, Optional[Dataset]]:
    """Load training and optional evaluation datasets.
    
    Evaluation dataset can come from:
    1. Separate eval_dataset_path
    2. Split from training data using val_size
    3. None if neither is specified
    
    Args:
        data_args: Data configuration arguments
    
    Returns:
        Tuple of (train_dataset, eval_dataset or None)
    """
    # Load main training dataset
    train_dataset = load_sft_dataset(data_args)
    eval_dataset = None
    
    # Check for separate eval dataset
    if data_args.eval_dataset_path:
        eval_dataset = load_local_dataset(data_args.eval_dataset_path)
        print(f"Loaded {len(eval_dataset)} eval examples from {data_args.eval_dataset_path}")
    
    # Or split from training data
    elif data_args.val_size > 0:
        if data_args.val_size >= 1.0:
            raise ValueError("val_size must be between 0 and 1")
        
        # Split the dataset
        split = train_dataset.train_test_split(
            test_size=data_args.val_size,
            seed=42,
        )
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Split dataset: {len(train_dataset)} train, {len(eval_dataset)} eval (val_size={data_args.val_size})")
    
    return train_dataset, eval_dataset


def prepare_dataset(
    dataset: Dataset,
    processor: SFTDatasetProcessor,
    num_workers: int = 4,
) -> Dataset:
    """Process and filter the dataset for training."""
    # Process dataset
    processed_dataset = dataset.map(
        processor.process_example,
        remove_columns=dataset.column_names,
        num_proc=num_workers,
        desc="Processing dataset",
    )
    
    # Filter empty examples
    processed_dataset = processed_dataset.filter(
        lambda x: len(x["input_ids"]) > 0,
        desc="Filtering empty examples",
    )
    
    return processed_dataset

