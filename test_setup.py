#!/usr/bin/env python3
"""
Test script to verify the Hugging Face Model Runner setup
"""

import sys
import os

def test_imports():
    """Test if all required packages can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Transformers import failed: {e}")
        return False
    
    try:
        import PIL
        print(f"✅ Pillow imported successfully")
    except ImportError as e:
        print(f"❌ Pillow import failed: {e}")
        return False
    
    try:
        import numpy
        print(f"✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    return True


def test_local_imports():
    """Test if local modules can be imported."""
    print("\n🔍 Testing local imports...")
    
    try:
        from config import ModelConfig
        print("✅ Config module imported successfully")
    except ImportError as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from main import HuggingFaceModelRunner
        print("✅ Main module imported successfully")
    except ImportError as e:
        print(f"❌ Main module import failed: {e}")
        return False
    
    return True


def test_device_detection():
    """Test device detection."""
    print("\n🔍 Testing device detection...")
    
    try:
        import torch
        from config import ModelConfig
        
        device_info = ModelConfig.get_device_info()
        print(f"✅ Device detected: {device_info}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("ℹ️  CUDA not available, will use CPU")
            
    except Exception as e:
        print(f"❌ Device detection failed: {e}")
        return False
    
    return True


def test_model_runner_initialization():
    """Test if the model runner can be initialized."""
    print("\n🔍 Testing model runner initialization...")
    
    try:
        from main import HuggingFaceModelRunner
        
        runner = HuggingFaceModelRunner()
        print("✅ Model runner initialized successfully")
        print(f"   Cache directory: {runner.cache_dir}")
        print(f"   Device: {runner.device}")
        
    except Exception as e:
        print(f"❌ Model runner initialization failed: {e}")
        return False
    
    return True


def test_configuration():
    """Test configuration settings."""
    print("\n🔍 Testing configuration...")
    
    try:
        from config import ModelConfig
        
        # Test model names
        text_gen_model = ModelConfig.get_model_name("text_generation")
        print(f"✅ Text generation model: {text_gen_model}")
        
        # Test model parameters
        params = ModelConfig.get_model_params("text_generation")
        print(f"✅ Text generation parameters: {params}")
        
        # Test device configuration
        device = ModelConfig.DEVICE
        print(f"✅ Device configuration: {device}")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("🚀 Hugging Face Model Runner - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_local_imports,
        test_device_detection,
        test_model_runner_initialization,
        test_configuration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your setup is ready to use.")
        print("\nNext steps:")
        print("1. Run 'python main.py' to see the demo")
        print("2. Try 'python examples/text_generation.py' for text generation")
        print("3. Check the README.md for more examples")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: pip install -r requirements.txt")
        print("2. Check your Python version (requires 3.8+)")
        print("3. Verify your internet connection for model downloads")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 