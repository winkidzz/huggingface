#!/usr/bin/env python3
"""
Image Classification Example
This script demonstrates how to use Hugging Face models for image classification.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HuggingFaceModelRunner
from config import ModelConfig
import requests
from PIL import Image
from io import BytesIO


def image_classification_example():
    """Example of image classification using different models."""
    print("🖼️  Image Classification Examples")
    print("=" * 40)
    
    # Initialize the model runner
    runner = HuggingFaceModelRunner()
    
    # Sample images for testing
    sample_images = [
        "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=400",  # Apple
        "https://images.unsplash.com/photo-1565299624946-b28f40a0ca4b?w=400",  # Pizza
        "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=400",  # Person
        "https://images.unsplash.com/photo-1441974231531-c6227db76b6e?w=400",  # Forest
        "https://images.unsplash.com/photo-1506905925346-21bda4d32df4?w=400"   # Mountain
    ]
    
    print("\n1️⃣ Basic Image Classification (ViT):")
    for i, image_url in enumerate(sample_images[:3], 1):
        try:
            result = runner.classify_image(image_url)
            print(f"{i}. Image: {image_url}")
            print(f"   Top prediction: {result['predictions'][0]['label']} "
                  f"(Confidence: {result['predictions'][0]['confidence']:.3f})")
            print()
        except Exception as e:
            print(f"{i}. Error processing image: {e}")
    
    # Example 2: Using a different image classification model
    print("\n2️⃣ Using a Different Image Model:")
    try:
        # Try loading a different model
        custom_model = "microsoft/resnet-50"
        runner.load_image_classification_model(custom_model)
        
        for i, image_url in enumerate(sample_images[3:], 4):
            try:
                result = runner.classify_image(image_url)
                print(f"{i}. Image: {image_url}")
                print(f"   Top prediction: {result['predictions'][0]['label']} "
                      f"(Confidence: {result['predictions'][0]['confidence']:.3f})")
                print()
            except Exception as e:
                print(f"{i}. Error processing image: {e}")
                
    except Exception as e:
        print(f"Could not load custom model: {e}")
    
    # Example 3: Detailed analysis of a single image
    print("\n3️⃣ Detailed Image Analysis:")
    test_image = sample_images[0]  # Apple image
    
    try:
        result = runner.classify_image(test_image)
        print(f"Image: {test_image}")
        print("Top 5 predictions:")
        for i, pred in enumerate(result['predictions'][:5], 1):
            print(f"  {i}. {pred['label']} ({pred['confidence']:.3f})")
    except Exception as e:
        print(f"Error in detailed analysis: {e}")
    
    print("\n✅ Image classification examples completed!")


def interactive_image_classification():
    """Interactive image classification session."""
    print("\n🎮 Interactive Image Classification")
    print("=" * 40)
    print("Enter image URLs or file paths. Type 'quit' to exit")
    
    runner = HuggingFaceModelRunner()
    
    while True:
        try:
            image_input = input("\nEnter image URL or file path: ").strip()
            
            if image_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if not image_input:
                continue
            
            # Classify the image
            result = runner.classify_image(image_input)
            
            print(f"\nClassification Results:")
            print("Top 3 predictions:")
            for i, pred in enumerate(result['predictions'][:3], 1):
                print(f"  {i}. {pred['label']} ({pred['confidence']:.3f})")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def compare_image_models():
    """Compare different image classification models."""
    print("\n🔍 Image Model Comparison")
    print("=" * 40)
    
    test_image = "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=400"
    
    models_to_test = [
        "google/vit-base-patch16-224",
        "microsoft/resnet-50",
        "facebook/deit-base-distilled-patch16-224"
    ]
    
    runner = HuggingFaceModelRunner()
    
    for model_name in models_to_test:
        try:
            print(f"\nTesting model: {model_name}")
            runner.load_image_classification_model(model_name)
            result = runner.classify_image(test_image)
            print(f"Top prediction: {result['predictions'][0]['label']} "
                  f"({result['predictions'][0]['confidence']:.3f})")
        except Exception as e:
            print(f"Error with model {model_name}: {e}")


def download_and_classify_image(image_url: str, save_path: str = None):
    """Download an image and classify it."""
    print(f"\n📥 Downloading and classifying image: {image_url}")
    
    try:
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Save image if path provided
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Image saved to: {save_path}")
        
        # Classify the image
        runner = HuggingFaceModelRunner()
        result = runner.classify_image(image_url)
        
        print("Classification results:")
        for i, pred in enumerate(result['predictions'][:5], 1):
            print(f"  {i}. {pred['label']} ({pred['confidence']:.3f})")
            
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "interactive":
            interactive_image_classification()
        elif command == "compare":
            compare_image_models()
        elif command == "download" and len(sys.argv) > 2:
            image_url = sys.argv[2]
            save_path = sys.argv[3] if len(sys.argv) > 3 else None
            download_and_classify_image(image_url, save_path)
        else:
            print("Unknown command. Available: interactive, compare, download <url> [save_path]")
    else:
        image_classification_example() 