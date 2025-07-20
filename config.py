"""
Configuration settings for Hugging Face models.
"""

import os
import torch
from typing import Dict, Any

class ModelConfig:
    """Configuration class for model settings."""
    
    # Default model settings
    DEFAULT_MODELS = {
        "text_generation": "gpt2",
        "text_classification": "distilbert-base-uncased-finetuned-sst-2-english",
        "translation": "t5-small",
        "image_classification": "google/vit-base-patch16-224",
        "summarization": "facebook/bart-large-cnn",
        "question_answering": "distilbert-base-cased-distilled-squad"
    }
    
    # Model parameters
    MODEL_PARAMS = {
        "text_generation": {
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        },
        "text_classification": {
            "return_all_scores": False
        },
        "translation": {
            "max_length": 128
        },
        "image_classification": {
            "top_k": 5
        },
        "summarization": {
            "max_length": 130,
            "min_length": 30
        },
        "question_answering": {
            "max_answer_length": 30
        }
    }
    
    # Device configuration
    DEVICE = "cuda" if os.environ.get("USE_CUDA", "true").lower() == "true" and torch.cuda.is_available() else "cpu"
    
    # Cache directory for models
    CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./models_cache")
    
    # Batch size for processing
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
    
    # Model loading settings
    LOAD_IN_8BIT = os.environ.get("LOAD_IN_8BIT", "false").lower() == "true"
    LOAD_IN_4BIT = os.environ.get("LOAD_IN_4BIT", "false").lower() == "true"
    
    @classmethod
    def get_model_name(cls, task: str) -> str:
        """Get the default model name for a given task."""
        return cls.DEFAULT_MODELS.get(task, "gpt2")
    
    @classmethod
    def get_model_params(cls, task: str) -> Dict[str, Any]:
        """Get the default parameters for a given task."""
        return cls.MODEL_PARAMS.get(task, {})
    
    @classmethod
    def get_device_info(cls) -> str:
        """Get information about the current device."""
        import torch
        if cls.DEVICE == "cuda" and torch.cuda.is_available():
            return f"CUDA ({torch.cuda.get_device_name(0)})"
        else:
            return "CPU" 