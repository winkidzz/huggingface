#!/usr/bin/env python3
"""
Translation Example
This script demonstrates how to use Hugging Face models for text translation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HuggingFaceModelRunner
from config import ModelConfig


def translation_example():
    """Example of translation using different models."""
    print("🌐 Translation Examples")
    print("=" * 40)
    
    # Initialize the model runner
    runner = HuggingFaceModelRunner()
    
    # Example texts for translation
    test_texts = [
        "Hello, how are you today?",
        "The weather is beautiful today.",
        "I love learning new languages.",
        "This is a wonderful place to visit.",
        "Thank you for your help."
    ]
    
    print("\n1️⃣ English to French Translation:")
    for i, text in enumerate(test_texts, 1):
        result = runner.translate_text(text)
        print(f"{i}. English: {text}")
        print(f"   French: {result}")
        print()
    
    # Example 2: Using a different translation model
    print("\n2️⃣ Using a Different Translation Model:")
    try:
        # Try loading a different model for English to Spanish
        custom_model = "Helsinki-NLP/opus-mt-en-es"
        runner.load_translation_model(custom_model)
        
        spanish_texts = [
            "Good morning, how are you?",
            "The food is delicious.",
            "I want to travel to Spain."
        ]
        
        for i, text in enumerate(spanish_texts, 1):
            result = runner.translate_text(text)
            print(f"{i}. English: {text}")
            print(f"   Spanish: {result}")
            print()
            
    except Exception as e:
        print(f"Could not load custom model: {e}")
    
    # Example 3: Batch translation
    print("\n3️⃣ Batch Translation:")
    batch_texts = [
        "Welcome to our website.",
        "Please contact us for more information.",
        "We hope you enjoy your stay."
    ]
    
    print("Batch translation results:")
    for text in batch_texts:
        result = runner.translate_text(text)
        print(f"'{text}' → '{result}'")
    
    print("\n✅ Translation examples completed!")


def interactive_translation():
    """Interactive translation session."""
    print("\n🎮 Interactive Translation")
    print("=" * 40)
    print("Type 'quit' to exit")
    
    runner = HuggingFaceModelRunner()
    
    while True:
        try:
            text = input("\nEnter text to translate: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Translate the text
            result = runner.translate_text(text)
            
            print(f"\nTranslation: {result}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def compare_translation_models():
    """Compare different translation models."""
    print("\n🔍 Translation Model Comparison")
    print("=" * 40)
    
    test_text = "Hello, how are you today?"
    
    models_to_test = [
        ("Helsinki-NLP/opus-mt-en-fr", "English to French"),
        ("Helsinki-NLP/opus-mt-en-es", "English to Spanish"),
        ("Helsinki-NLP/opus-mt-en-de", "English to German")
    ]
    
    runner = HuggingFaceModelRunner()
    
    for model_name, description in models_to_test:
        try:
            print(f"\nTesting {description}: {model_name}")
            runner.load_translation_model(model_name)
            result = runner.translate_text(test_text)
            print(f"English: {test_text}")
            print(f"Translation: {result}")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")


def multi_language_translation():
    """Demonstrate translation to multiple languages."""
    print("\n🌍 Multi-Language Translation")
    print("=" * 40)
    
    text = "The artificial intelligence revolution is transforming our world."
    
    language_models = [
        ("Helsinki-NLP/opus-mt-en-fr", "French"),
        ("Helsinki-NLP/opus-mt-en-es", "Spanish"),
        ("Helsinki-NLP/opus-mt-en-de", "German"),
        ("Helsinki-NLP/opus-mt-en-it", "Italian"),
        ("Helsinki-NLP/opus-mt-en-pt", "Portuguese")
    ]
    
    runner = HuggingFaceModelRunner()
    
    print(f"Original (English): {text}")
    print()
    
    for model_name, language in language_models:
        try:
            runner.load_translation_model(model_name)
            result = runner.translate_text(text)
            print(f"{language}: {result}")
        except Exception as e:
            print(f"{language}: Error - {e}")


def translation_with_context():
    """Demonstrate translation with different contexts."""
    print("\n📝 Contextual Translation")
    print("=" * 40)
    
    runner = HuggingFaceModelRunner()
    
    # Different contexts for the same word
    contexts = [
        "I will bank the money tomorrow.",  # Financial context
        "The bank of the river is muddy.",  # Geographic context
        "I need to bank left at the intersection."  # Aviation context
    ]
    
    print("Translating 'bank' in different contexts:")
    for context in contexts:
        result = runner.translate_text(context)
        print(f"Context: {context}")
        print(f"Translation: {result}")
        print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            interactive_translation()
        elif command == "compare":
            compare_translation_models()
        elif command == "multi":
            multi_language_translation()
        elif command == "context":
            translation_with_context()
        else:
            print("Unknown command. Available: interactive, compare, multi, context")
    else:
        translation_example() 