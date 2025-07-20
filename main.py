#!/usr/bin/env python3
"""
Main script for running Hugging Face models locally.
This script provides examples of loading and using different types of models.
"""

import os
import sys
from typing import Optional, Dict, Any
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM, AutoModelForImageClassification, AutoProcessor,
    pipeline
)
from PIL import Image
import requests
from io import BytesIO

from config import ModelConfig


class HuggingFaceModelRunner:
    """Main class for running Hugging Face models."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the model runner."""
        self.cache_dir = cache_dir or ModelConfig.CACHE_DIR
        self.device = ModelConfig.DEVICE
        self.models = {}
        self.tokenizers = {}
        self.processors = {}
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"🚀 Initializing Hugging Face Model Runner")
        print(f"📁 Cache directory: {self.cache_dir}")
        print(f"🖥️  Device: {ModelConfig.get_device_info()}")
    
    def load_text_generation_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Load a text generation model."""
        model_name = model_name or ModelConfig.get_model_name("text_generation")
        print(f"📝 Loading text generation model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.models["text_generation"] = model
            self.tokenizers["text_generation"] = tokenizer
            
            return {"model": model, "tokenizer": tokenizer}
            
        except Exception as e:
            print(f"❌ Error loading text generation model: {e}")
            return {}
    
    def load_text_classification_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Load a text classification model."""
        model_name = model_name or ModelConfig.get_model_name("text_classification")
        print(f"🏷️  Loading text classification model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir
            )
            
            model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.models["text_classification"] = model
            self.tokenizers["text_classification"] = tokenizer
            
            return {"model": model, "tokenizer": tokenizer}
            
        except Exception as e:
            print(f"❌ Error loading text classification model: {e}")
            return {}
    
    def load_translation_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Load a translation model."""
        model_name = model_name or ModelConfig.get_model_name("translation")
        print(f"🌐 Loading translation model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir
            )
            
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.models["translation"] = model
            self.tokenizers["translation"] = tokenizer
            
            return {"model": model, "tokenizer": tokenizer}
            
        except Exception as e:
            print(f"❌ Error loading translation model: {e}")
            return {}
    
    def load_image_classification_model(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Load an image classification model."""
        model_name = model_name or ModelConfig.get_model_name("image_classification")
        print(f"🖼️  Loading image classification model: {model_name}")
        
        try:
            processor = AutoProcessor.from_pretrained(
                model_name, 
                cache_dir=self.cache_dir
            )
            
            model = AutoModelForImageClassification.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            if self.device == "cpu":
                model = model.to(self.device)
            
            self.models["image_classification"] = model
            self.processors["image_classification"] = processor
            
            return {"model": model, "processor": processor}
            
        except Exception as e:
            print(f"❌ Error loading image classification model: {e}")
            return {}
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the loaded text generation model."""
        if "text_generation" not in self.models:
            print("❌ Text generation model not loaded. Loading default model...")
            self.load_text_generation_model()
        
        model = self.models["text_generation"]
        tokenizer = self.tokenizers["text_generation"]
        
        # Get default parameters
        params = ModelConfig.get_model_params("text_generation")
        params.update(kwargs)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **params
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def classify_text(self, text: str) -> Dict[str, Any]:
        """Classify text using the loaded classification model."""
        if "text_classification" not in self.models:
            print("❌ Text classification model not loaded. Loading default model...")
            self.load_text_classification_model()
        
        model = self.models["text_classification"]
        tokenizer = self.tokenizers["text_classification"]
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Get label mapping
        if hasattr(model.config, 'id2label'):
            label_mapping = model.config.id2label
        else:
            label_mapping = {0: "NEGATIVE", 1: "POSITIVE"}
        
        # Get top prediction
        top_prob, top_class = torch.max(probabilities, dim=-1)
        
        return {
            "label": label_mapping[top_class.item()],
            "confidence": top_prob.item(),
            "probabilities": {label_mapping[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        }
    
    def translate_text(self, text: str, **kwargs) -> str:
        """Translate text using the loaded translation model."""
        if "translation" not in self.models:
            print("❌ Translation model not loaded. Loading default model...")
            self.load_translation_model()
        
        model = self.models["translation"]
        tokenizer = self.tokenizers["translation"]
        
        # Get default parameters
        params = ModelConfig.get_model_params("translation")
        params.update(kwargs)
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, **params)
        
        translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """Classify an image using the loaded image classification model."""
        if "image_classification" not in self.models:
            print("❌ Image classification model not loaded. Loading default model...")
            self.load_image_classification_model()
        
        model = self.models["image_classification"]
        processor = self.processors["image_classification"]
        
        # Load and preprocess image
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content))
        else:
            image = Image.open(image_path)
        
        inputs = processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get top predictions
        top_k = ModelConfig.get_model_params("image_classification").get("top_k", 5)
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
        # Get label mapping
        if hasattr(model.config, 'id2label'):
            label_mapping = model.config.id2label
        else:
            label_mapping = {i: f"class_{i}" for i in range(model.config.num_labels)}
        
        results = []
        for prob, idx in zip(top_probs[0], top_indices[0]):
            results.append({
                "label": label_mapping[idx.item()],
                "confidence": prob.item()
            })
        
        return {"predictions": results}
    
    def run_demo(self):
        """Run a demonstration of all model types."""
        print("\n🎯 Running Hugging Face Models Demo")
        print("=" * 50)
        
        # Text Generation Demo
        print("\n📝 Text Generation Demo:")
        prompt = "The future of artificial intelligence is"
        generated_text = self.generate_text(prompt)
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated_text}")
        
        # Text Classification Demo
        print("\n🏷️  Text Classification Demo:")
        text_to_classify = "I love this movie, it's absolutely fantastic!"
        classification = self.classify_text(text_to_classify)
        print(f"Text: {text_to_classify}")
        print(f"Classification: {classification}")
        
        # Translation Demo
        print("\n🌐 Translation Demo:")
        text_to_translate = "Hello, how are you today?"
        translated = self.translate_text(text_to_translate)
        print(f"English: {text_to_translate}")
        print(f"French: {translated}")
        
        # Image Classification Demo (using a sample image)
        print("\n🖼️  Image Classification Demo:")
        try:
            # Use a sample image from the internet
            sample_image_url = "https://images.unsplash.com/photo-1541963463532-d68292c34b19?w=400"
            image_classification = self.classify_image(sample_image_url)
            print(f"Sample image: {sample_image_url}")
            print(f"Classification: {image_classification}")
        except Exception as e:
            print(f"Image classification failed: {e}")
        
        print("\n✅ Demo completed!")


def main():
    """Main function to run the Hugging Face model examples."""
    print("🤗 Welcome to Hugging Face Model Runner!")
    
    # Initialize the model runner
    runner = HuggingFaceModelRunner()
    
    # Check command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "demo":
            runner.run_demo()
        elif command == "text-gen":
            if len(sys.argv) > 2:
                prompt = sys.argv[2]
                result = runner.generate_text(prompt)
                print(f"Generated text: {result}")
            else:
                print("Usage: python main.py text-gen <prompt>")
        elif command == "classify":
            if len(sys.argv) > 2:
                text = sys.argv[2]
                result = runner.classify_text(text)
                print(f"Classification: {result}")
            else:
                print("Usage: python main.py classify <text>")
        elif command == "translate":
            if len(sys.argv) > 2:
                text = sys.argv[2]
                result = runner.translate_text(text)
                print(f"Translation: {result}")
            else:
                print("Usage: python main.py translate <text>")
        else:
            print("Unknown command. Available commands: demo, text-gen, classify, translate")
    else:
        # Run demo by default
        runner.run_demo()


if __name__ == "__main__":
    main() 