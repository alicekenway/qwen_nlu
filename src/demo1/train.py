"""
Training script for Qwen2 SFT with LoRA.

This script handles the training loop and model saving.

Supports two modes:
1. CLI arguments: python train.py --model_name_or_path ... --output_dir ...
2. YAML config:   python train.py config.yaml [--override_arg value ...]
"""

import sys
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from transformers import HfArgumentParser, Trainer
from trl import SFTConfig

from data import (
    IGNORE_INDEX,
    DataArguments,
    Qwen2Template,
    SFTDataCollator,
    SFTDatasetProcessor,
    load_train_eval_datasets,
    prepare_dataset,
)
from model import LoraArguments, ModelArguments, load_model_and_tokenizer


# =============================================================================
# Argument Parsing with YAML Support
# =============================================================================
def load_yaml_config(yaml_path: str) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config if config else {}


def parse_cli_overrides(args: list[str]) -> dict[str, Any]:
    """Parse CLI arguments as overrides (--key value format).
    
    Handles various value types:
    - Booleans: true/false, True/False
    - Numbers: int and float
    - null/None: None
    - Strings: everything else
    """
    overrides = {}
    i = 0
    while i < len(args):
        if args[i].startswith('--'):
            key = args[i][2:]  # Remove '--'
            if i + 1 < len(args) and not args[i + 1].startswith('--'):
                value = args[i + 1]
                # Type conversion
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.lower() in ('null', 'none'):
                    value = None
                else:
                    # Try numeric conversion
                    try:
                        if '.' in value or 'e' in value.lower():
                            value = float(value)
                        else:
                            value = int(value)
                    except ValueError:
                        pass  # Keep as string
                overrides[key] = value
                i += 2
            else:
                # Flag without value (treat as True)
                overrides[key] = True
                i += 1
        else:
            i += 1
    return overrides


def read_args() -> Union[dict[str, Any], list[str]]:
    """Read arguments from CLI or YAML config file.
    
    If the first argument is a .yaml or .yml file, load it and merge
    with any CLI overrides. Otherwise, return CLI args as-is.
    
    Usage:
        python train.py config.yaml                    # Load from YAML
        python train.py config.yaml --lr 1e-4          # YAML + overrides
        python train.py --model_name_or_path ...       # Direct CLI args
    """
    if len(sys.argv) < 2:
        return []
    
    first_arg = sys.argv[1]
    
    # Check if first argument is a YAML file
    if first_arg.endswith('.yaml') or first_arg.endswith('.yml'):
        yaml_path = Path(first_arg).absolute()
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        print(f"Loading config from: {yaml_path}")
        config = load_yaml_config(str(yaml_path))
        
        # Parse CLI overrides (arguments after the YAML file)
        if len(sys.argv) > 2:
            overrides = parse_cli_overrides(sys.argv[2:])
            if overrides:
                print(f"CLI overrides: {overrides}")
                config.update(overrides)
        
        return config
    else:
        # Standard CLI arguments
        return sys.argv[1:]


def parse_args(
    args: Optional[Union[dict[str, Any], list[str]]] = None
) -> tuple[ModelArguments, LoraArguments, DataArguments, SFTConfig]:
    """Parse arguments from dict (YAML) or list (CLI).
    
    Args:
        args: Either a dict from YAML or list of CLI args.
              If None, reads from command line.
    
    Returns:
        Tuple of (ModelArguments, LoraArguments, DataArguments, SFTConfig)
    """
    parser = HfArgumentParser((ModelArguments, LoraArguments, DataArguments, SFTConfig))
    
    if args is None:
        args = read_args()
    
    if isinstance(args, dict):
        # Parse from dictionary (YAML config)
        model_args, lora_args, data_args, training_args = parser.parse_dict(args)
    else:
        # Parse from command line arguments
        model_args, lora_args, data_args, training_args = parser.parse_args_into_dataclasses(args)
    
    return model_args, lora_args, data_args, training_args


# =============================================================================
# Training Utilities
# =============================================================================
def print_sample(dataset, tokenizer, num_samples: int = 1):
    """Print sample data for verification."""
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n=== Sample {i + 1} ===")
        print(f"input_ids length: {len(sample['input_ids'])}")
        print(f"labels length: {len(sample['labels'])}")
        
        # Decode and show
        valid_labels = [l for l in sample['labels'] if l != IGNORE_INDEX]
        print(f"\nFull input:\n{tokenizer.decode(sample['input_ids'], skip_special_tokens=False)}")
        print(f"\nLabels (what model learns):\n{tokenizer.decode(valid_labels, skip_special_tokens=False)}")


def print_config(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    data_args: DataArguments,
    training_args: SFTConfig,
):
    """Print configuration summary."""
    print("\n" + "=" * 60)
    print("CONFIGURATION SUMMARY")
    print("=" * 60)
    
    print("\n[Model]")
    print(f"  model_name_or_path: {model_args.model_name_or_path}")
    print(f"  use_4bit: {model_args.use_4bit}")
    print(f"  use_8bit: {model_args.use_8bit}")
    print(f"  use_flash_attention: {model_args.use_flash_attention}")
    
    print("\n[LoRA]")
    print(f"  use_lora: {lora_args.use_lora}")
    if lora_args.use_lora:
        print(f"  lora_r: {lora_args.lora_r}")
        print(f"  lora_alpha: {lora_args.lora_alpha}")
        print(f"  lora_dropout: {lora_args.lora_dropout}")
        print(f"  target_modules: {lora_args.target_modules}")
        print(f"  layers_to_transform: {lora_args.layers_to_transform or 'all'}")
    else:
        print("  (Full fine-tuning mode)")
    
    print("\n[Data]")
    print(f"  dataset_path: {data_args.dataset_path}")
    print(f"  dataset_name: {data_args.dataset_name}")
    print(f"  eval_dataset_path: {data_args.eval_dataset_path}")
    print(f"  val_size: {data_args.val_size}")
    print(f"  max_seq_length: {data_args.max_seq_length}")
    
    print("\n[Training]")
    print(f"  output_dir: {training_args.output_dir}")
    print(f"  num_train_epochs: {training_args.num_train_epochs}")
    print(f"  per_device_train_batch_size: {training_args.per_device_train_batch_size}")
    print(f"  gradient_accumulation_steps: {training_args.gradient_accumulation_steps}")
    print(f"  learning_rate: {training_args.learning_rate}")
    print(f"  lr_scheduler_type: {training_args.lr_scheduler_type}")
    print(f"  warmup_ratio: {training_args.warmup_ratio}")
    print(f"  bf16: {training_args.bf16}")
    print(f"  fp16: {training_args.fp16}")
    
    print("\n[Evaluation]")
    eval_strategy = getattr(training_args, 'eval_strategy', getattr(training_args, 'evaluation_strategy', 'no'))
    print(f"  eval_strategy: {eval_strategy}")
    if eval_strategy != 'no':
        print(f"  eval_steps: {training_args.eval_steps}")
        print(f"  per_device_eval_batch_size: {training_args.per_device_eval_batch_size}")
    
    print("\n[Logging]")
    print(f"  logging_steps: {training_args.logging_steps}")
    print(f"  report_to: {training_args.report_to}")
    if 'tensorboard' in str(training_args.report_to):
        print(f"  logging_dir: {training_args.logging_dir}")
    
    print("=" * 60 + "\n")


# =============================================================================
# Main Training Function
# =============================================================================
def train(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    data_args: DataArguments,
    training_args: SFTConfig,
):
    """Run the training loop with optional evaluation."""
    # Print configuration
    print_config(model_args, lora_args, data_args, training_args)
    
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
    
    # Load train and eval datasets
    print("Loading datasets...")
    train_dataset, eval_dataset = load_train_eval_datasets(data_args)
    
    # Print sample before processing
    print("\n=== Sample raw data ===")
    sample = train_dataset[0]
    for key, value in sample.items():
        print(f"{key}: {str(value)[:100]}...")
    
    # Process training dataset
    print("\nProcessing training dataset...")
    processed_train = prepare_dataset(
        dataset=train_dataset,
        processor=processor,
        num_workers=data_args.preprocessing_num_workers,
    )
    print(f"Processed {len(processed_train)} training examples")
    
    # Process eval dataset if available
    processed_eval = None
    if eval_dataset is not None:
        print("\nProcessing evaluation dataset...")
        processed_eval = prepare_dataset(
            dataset=eval_dataset,
            processor=processor,
            num_workers=data_args.preprocessing_num_workers,
        )
        print(f"Processed {len(processed_eval)} evaluation examples")
    
    # Print sample after processing
    print_sample(processed_train, tokenizer)
    
    # Create data collator
    data_collator = SFTDataCollator(
        tokenizer=tokenizer,
        max_seq_length=data_args.max_seq_length,
    )
    
    # Create trainer with optional eval dataset
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_train,
        eval_dataset=processed_eval,
        data_collator=data_collator,
        processing_class=tokenizer,
    )
    
    # Train
    print("\n=== Starting training ===")
    if processed_eval is not None:
        eval_strategy = getattr(training_args, 'eval_strategy', getattr(training_args, 'evaluation_strategy', 'no'))
        print(f"Evaluation enabled: strategy={eval_strategy}, steps={training_args.eval_steps}")
    trainer.train()
    
    # Final evaluation
    if processed_eval is not None:
        print("\n=== Final Evaluation ===")
        eval_results = trainer.evaluate()
        print(f"Eval results: {eval_results}")
    
    # Save
    print(f"\nSaving model to {training_args.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    print("\n=== Training complete! ===")


def main():
    """Main entry point."""
    # Parse arguments (from YAML or CLI)
    model_args, lora_args, data_args, training_args = parse_args()
    
    # Run training
    train(model_args, lora_args, data_args, training_args)


if __name__ == "__main__":
    main()
