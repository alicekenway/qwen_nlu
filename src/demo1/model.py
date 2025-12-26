"""
Model module for Qwen2 SFT training.

Contains model wrapper class and configuration for LoRA fine-tuning.
Designed to support future modifications and custom loss functions.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)


# =============================================================================
# Model Arguments
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
    use_lora: bool = field(
        default=True,
        metadata={"help": "Whether to use LoRA. If False, full fine-tuning is used"}
    )
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})
    target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated target modules for LoRA"}
    )
    layers_to_transform: Optional[str] = field(
        default=None,
        metadata={"help": "Comma-separated layer indices or ranges to apply LoRA (e.g., '0,1,2,3' or '0-3' or '0,1,20-23'). None = all layers"}
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Layer name pattern for matching (advanced use, e.g., 'layers')"}
    )


# =============================================================================
# Configuration Helpers
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


def parse_layers_to_transform(layers_str: Optional[str]) -> Optional[list[int]]:
    """Parse layers string to list of integers.
    
    Supports:
    - Comma-separated: "0,1,2,3"
    - Range: "0-3" (inclusive)
    - Mixed: "0,1,5-10,15"
    
    Args:
        layers_str: String representation of layers
    
    Returns:
        Sorted list of unique layer indices, or None if input is None/empty
    """
    if not layers_str:
        return None
    
    layers = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            # Range format: "0-3"
            start, end = part.split("-")
            layers.extend(range(int(start), int(end) + 1))
        else:
            layers.append(int(part))
    
    return sorted(set(layers))


def create_lora_config(lora_args: LoraArguments) -> LoraConfig:
    """Create LoRA configuration."""
    target_modules = lora_args.target_modules.split(",") if lora_args.target_modules else None
    layers_to_transform = parse_layers_to_transform(lora_args.layers_to_transform)
    
    return LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        target_modules=target_modules,
        layers_to_transform=layers_to_transform,
        layers_pattern=lora_args.layers_pattern,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


# =============================================================================
# Model Wrapper Class
# =============================================================================
class Qwen2SFTModel(nn.Module):
    """Wrapper class for Qwen2 model with LoRA.
    
    This wrapper allows for:
    - Custom loss functions
    - Additional forward logic
    - Easy extension for future modifications
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        custom_loss_fn: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = base_model
        self.tokenizer = tokenizer
        self.custom_loss_fn = custom_loss_fn
        self.config = base_model.config
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        """Forward pass with optional custom loss computation."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )
        
        # Apply custom loss if provided
        if self.custom_loss_fn is not None and labels is not None:
            logits = outputs.logits
            custom_loss = self.custom_loss_fn(logits, labels)
            # Replace the loss
            outputs.loss = custom_loss
        
        return outputs
    
    def generate(self, *args, **kwargs):
        """Delegate generation to the underlying model."""
        return self.model.generate(*args, **kwargs)
    
    def save_pretrained(self, output_dir: str, **kwargs):
        """Save the model."""
        self.model.save_pretrained(output_dir, **kwargs)
    
    def print_trainable_parameters(self):
        """Print trainable parameters info."""
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing."""
        self.model.gradient_checkpointing_enable()
    
    @property
    def device(self):
        """Get the device of the model."""
        return next(self.parameters()).device


# =============================================================================
# Model Loading Functions
# =============================================================================
def load_tokenizer(model_args: ModelArguments) -> PreTrainedTokenizer:
    """Load and configure tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_base_model(
    model_args: ModelArguments,
    bnb_config: Optional[BitsAndBytesConfig] = None,
) -> PreTrainedModel:
    """Load the base pretrained model."""
    # Determine attention implementation
    attn_impl = None
    if model_args.use_flash_attention and torch.cuda.is_available():
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
        except ImportError:
            print("Flash attention not available, using default attention")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16 if not bnb_config else None,
        attn_implementation=attn_impl,
    )
    return model


def load_model_and_tokenizer(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    use_wrapper: bool = False,
    custom_loss_fn: Optional[nn.Module] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer with optional LoRA configuration.
    
    Args:
        model_args: Model configuration arguments
        lora_args: LoRA configuration arguments (use_lora controls LoRA vs full fine-tuning)
        use_wrapper: If True, return Qwen2SFTModel wrapper instead of raw model
        custom_loss_fn: Optional custom loss function for the wrapper
    
    Returns:
        Tuple of (model, tokenizer)
    """
    bnb_config = create_bnb_config(model_args.use_4bit, model_args.use_8bit)
    
    # Load tokenizer
    tokenizer = load_tokenizer(model_args)
    
    # Load model
    model = load_base_model(model_args, bnb_config)
    model.gradient_checkpointing_enable()
    
    # Prepare for k-bit training if using quantization
    if model_args.use_4bit or model_args.use_8bit:
        model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA or full fine-tuning
    if lora_args.use_lora:
        lora_config = create_lora_config(lora_args)
        model = get_peft_model(model, lora_config)
        
        # Print layer info if layers_to_transform is specified
        if lora_args.layers_to_transform:
            layers = parse_layers_to_transform(lora_args.layers_to_transform)
            print(f"LoRA applied to layers: {layers}")
        else:
            print("LoRA applied to all layers")
        
        model.print_trainable_parameters()
    else:
        # Full fine-tuning: all parameters are trainable
        print("Full fine-tuning mode (no LoRA)")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    
    # Optionally wrap in custom model class
    if use_wrapper:
        model = Qwen2SFTModel(
            base_model=model,
            tokenizer=tokenizer,
            custom_loss_fn=custom_loss_fn,
        )
    
    return model, tokenizer


def load_model_for_inference(
    model_path: str,
    device: str = "auto",
    use_4bit: bool = False,
    use_8bit: bool = False,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a trained model for inference.
    
    Args:
        model_path: Path to the saved model (with LoRA adapters)
        device: Device to load model on
        use_4bit: Use 4-bit quantization
        use_8bit: Use 8-bit quantization
        trust_remote_code: Trust remote code
    
    Returns:
        Tuple of (model, tokenizer)
    """
    from peft import PeftModel
    
    bnb_config = create_bnb_config(use_4bit, use_8bit)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",  # Left padding for batch generation
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Try to load as PEFT model first
    try:
        # Load adapter config to get base model path
        from peft import PeftConfig
        peft_config = PeftConfig.from_pretrained(model_path)
        base_model_path = peft_config.base_model_name_or_path
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
        )
        
        # Load PEFT adapters
        model = PeftModel.from_pretrained(base_model, model_path)
        print(f"Loaded PEFT model from {model_path}")
        
    except Exception:
        # Fall back to loading as regular model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.bfloat16 if not bnb_config else None,
        )
        print(f"Loaded model from {model_path}")
    
    model.eval()
    return model, tokenizer

