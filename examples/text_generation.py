#!/usr/bin/env python3
"""
Text Generation Example
This script demonstrates how to use Hugging Face models for text generation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HuggingFaceModelRunner
from config import ModelConfig


def text_generation_example():
    """Example of text generation using different models."""
    print("📝 Text Generation Examples")
    print("=" * 40)
    
    # Initialize the model runner
    runner = HuggingFaceModelRunner()
    
    # Example 1: Basic text generation with GPT-2
    print("\n1️⃣ Basic Text Generation (GPT-2):")
    prompt = "The future of artificial intelligence is"
    result = runner.generate_text(prompt)
    print(f"Prompt: {prompt}")
    print(f"Generated: {result}")
    
    # Example 2: Creative writing
    print("\n2️⃣ Creative Writing:")
    creative_prompt = "Once upon a time in a magical forest"
    creative_result = runner.generate_text(
        creative_prompt,
        max_length=150,
        temperature=0.9,
        top_p=0.95
    )
    print(f"Prompt: {creative_prompt}")
    print(f"Generated: {creative_result}")
    
    # Example 3: Technical writing
    print("\n3️⃣ Technical Writing:")
    tech_prompt = "The benefits of machine learning include"
    tech_result = runner.generate_text(
        tech_prompt,
        max_length=120,
        temperature=0.5,
        top_p=0.8
    )
    print(f"Prompt: {tech_prompt}")
    print(f"Generated: {tech_result}")
    
    # Example 4: Custom model (if available)
    print("\n4️⃣ Using a Different Model:")
    try:
        # Try loading a different model
        custom_model = "microsoft/DialoGPT-medium"
        runner.load_text_generation_model(custom_model)
        
        custom_prompt = "Hello, how are you?"
        custom_result = runner.generate_text(custom_prompt)
        print(f"Model: {custom_model}")
        print(f"Prompt: {custom_prompt}")
        print(f"Generated: {custom_result}")
    except Exception as e:
        print(f"Could not load custom model: {e}")
    
    print("\n✅ Text generation examples completed!")


def interactive_text_generation():
    """Interactive text generation session."""
    print("\n🎮 Interactive Text Generation")
    print("=" * 40)
    print("Type 'quit' to exit")
    
    runner = HuggingFaceModelRunner()
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            # Get generation parameters
            max_length = input("Max length (default 100): ").strip()
            max_length = int(max_length) if max_length.isdigit() else 100
            
            temperature = input("Temperature (default 0.7): ").strip()
            temperature = float(temperature) if temperature.replace('.', '').isdigit() else 0.7
            
            # Generate text
            result = runner.generate_text(
                prompt,
                max_length=max_length,
                temperature=temperature
            )
            
            print(f"\nGenerated text: {result}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        interactive_text_generation()
    else:
        text_generation_example() 