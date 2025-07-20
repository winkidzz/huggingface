#!/usr/bin/env python3
"""
Text Classification Example
This script demonstrates how to use Hugging Face models for text classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HuggingFaceModelRunner
from config import ModelConfig


def text_classification_example():
    """Example of text classification using different models."""
    print("🏷️  Text Classification Examples")
    print("=" * 40)
    
    # Initialize the model runner
    runner = HuggingFaceModelRunner()
    
    # Example texts for classification
    test_texts = [
        "I love this movie, it's absolutely fantastic!",
        "This is the worst film I've ever seen.",
        "The weather is nice today.",
        "I'm feeling really sad and depressed.",
        "The food at this restaurant is delicious!",
        "This product is terrible, I want my money back.",
        "The concert was amazing, the music was incredible.",
        "I'm so disappointed with the service here.",
        "The book was well-written and engaging.",
        "This is a neutral statement about technology."
    ]
    
    print("\n1️⃣ Sentiment Analysis (Default Model):")
    for i, text in enumerate(test_texts[:5], 1):
        result = runner.classify_text(text)
        print(f"{i}. Text: {text}")
        print(f"   Classification: {result['label']} (Confidence: {result['confidence']:.3f})")
        print()
    
    # Example 2: Using a different classification model
    print("\n2️⃣ Using a Different Classification Model:")
    try:
        # Try loading a different model for more specific classification
        custom_model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        runner.load_text_classification_model(custom_model)
        
        for i, text in enumerate(test_texts[5:], 6):
            result = runner.classify_text(text)
            print(f"{i}. Text: {text}")
            print(f"   Classification: {result['label']} (Confidence: {result['confidence']:.3f})")
            print()
            
    except Exception as e:
        print(f"Could not load custom model: {e}")
    
    # Example 3: Batch classification
    print("\n3️⃣ Batch Classification:")
    batch_texts = [
        "The customer service was excellent!",
        "I'm very unhappy with this purchase.",
        "The quality is acceptable but not great.",
        "This exceeded all my expectations!"
    ]
    
    print("Batch processing results:")
    for text in batch_texts:
        result = runner.classify_text(text)
        print(f"'{text}' → {result['label']} ({result['confidence']:.3f})")
    
    print("\n✅ Text classification examples completed!")


def interactive_classification():
    """Interactive text classification session."""
    print("\n🎮 Interactive Text Classification")
    print("=" * 40)
    print("Type 'quit' to exit")
    
    runner = HuggingFaceModelRunner()
    
    while True:
        try:
            text = input("\nEnter text to classify: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text:
                continue
            
            # Classify the text
            result = runner.classify_text(text)
            
            print(f"\nClassification Results:")
            print(f"Label: {result['label']}")
            print(f"Confidence: {result['confidence']:.3f}")
            
            if 'probabilities' in result:
                print("All probabilities:")
                for label, prob in result['probabilities'].items():
                    print(f"  {label}: {prob:.3f}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def compare_models():
    """Compare different classification models."""
    print("\n🔍 Model Comparison")
    print("=" * 40)
    
    test_text = "I'm really happy with this product!"
    
    models_to_test = [
        "distilbert-base-uncased-finetuned-sst-2-english",
        "cardiffnlp/twitter-roberta-base-sentiment-latest"
    ]
    
    runner = HuggingFaceModelRunner()
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            runner.load_text_classification_model(model_name)
            result = runner.classify_text(test_text)
            print(f"Text: {test_text}")
            print(f"Result: {result['label']} (Confidence: {result['confidence']:.3f})")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            interactive_classification()
        elif command == "compare":
            compare_models()
        else:
            print("Unknown command. Available: interactive, compare")
    else:
        text_classification_example() 