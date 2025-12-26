"""
Inference script for Qwen2 SFT model.

This script handles loading a trained model and generating responses.
"""

import argparse
from typing import Optional

import torch

from data import Qwen2Template
from model import load_model_for_inference


def generate_response(
    model,
    tokenizer,
    template: Qwen2Template,
    user_input: str,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a response for the given user input.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        template: Qwen2Template for formatting
        user_input: User's input text
        system_prompt: Optional system prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        do_sample: Whether to use sampling
        repetition_penalty: Penalty for repetition
    
    Returns:
        Generated response text
    """
    # Format the prompt
    prompt = template.format_prompt(user_input, system=system_prompt)
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the generated part
    input_length = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][input_length:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up the response (remove any trailing special tokens)
    if "<|im_end|>" in response:
        response = response.split("<|im_end|>")[0]
    
    return response.strip()


def batch_generate(
    model,
    tokenizer,
    template: Qwen2Template,
    user_inputs: list[str],
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> list[str]:
    """Generate responses for multiple inputs in batch.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        template: Qwen2Template for formatting
        user_inputs: List of user inputs
        system_prompt: Optional system prompt (applied to all)
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        do_sample: Whether to use sampling
    
    Returns:
        List of generated responses
    """
    # Format all prompts
    prompts = [template.format_prompt(inp, system=system_prompt) for inp in user_inputs]
    
    # Tokenize with padding
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode responses
    responses = []
    for i, output in enumerate(outputs):
        # Find where the input ends (after padding)
        input_length = (inputs["attention_mask"][i] == 1).sum().item()
        generated_ids = output[input_length:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Clean up
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        responses.append(response.strip())
    
    return responses


def interactive_chat(
    model,
    tokenizer,
    template: Qwen2Template,
    system_prompt: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
):
    """Run an interactive chat session.
    
    Args:
        model: The loaded model
        tokenizer: The tokenizer
        template: Qwen2Template for formatting
        system_prompt: Optional system prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    print("\n=== Interactive Chat ===")
    print("Type 'quit' or 'exit' to end the session.")
    print("Type 'clear' to start a new conversation.")
    if system_prompt:
        print(f"System: {system_prompt}")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break
        
        if user_input.lower() == 'clear':
            print("\n--- Conversation cleared ---")
            continue
        
        # Generate response
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            template=template,
            user_input=user_input,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        print(f"\nAssistant: {response}")


def main():
    parser = argparse.ArgumentParser(description="Inference with trained Qwen2 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt for inference (if not provided, starts interactive mode)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p sampling parameter",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Use 4-bit quantization for inference",
    )
    parser.add_argument(
        "--use_8bit",
        action="store_true",
        help="Use 8-bit quantization for inference",
    )
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, tokenizer = load_model_for_inference(
        model_path=args.model_path,
        use_4bit=args.use_4bit,
        use_8bit=args.use_8bit,
    )
    
    # Create template
    template = Qwen2Template()
    
    if args.prompt:
        # Single inference
        print(f"\nPrompt: {args.prompt}")
        response = generate_response(
            model=model,
            tokenizer=tokenizer,
            template=template,
            user_input=args.prompt,
            system_prompt=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        print(f"\nResponse: {response}")
    else:
        # Interactive mode
        interactive_chat(
            model=model,
            tokenizer=tokenizer,
            template=template,
            system_prompt=args.system,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )


if __name__ == "__main__":
    main()

