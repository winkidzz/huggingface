#!/usr/bin/env python3
"""
MedGemma Setup Script
This script helps you set up MedGemma authentication and configuration.
"""

import os
import sys
import subprocess
import requests
from pathlib import Path


def check_huggingface_login():
    """Check if user is logged into Hugging Face."""
    try:
        # Try to get user info from Hugging Face
        result = subprocess.run(
            ["huggingface-cli", "whoami"], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        if result.returncode == 0:
            username = result.stdout.strip()
            print(f"✅ Logged in as: {username}")
            return True
        else:
            print("❌ Not logged into Hugging Face")
            return False
    except FileNotFoundError:
        print("❌ huggingface-cli not found. Please install it:")
        print("   pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"❌ Error checking login status: {e}")
        return False


def login_to_huggingface():
    """Guide user through Hugging Face login process."""
    print("\n🔐 Hugging Face Login Process")
    print("=" * 40)
    
    print("1. First, install huggingface_hub if not already installed:")
    print("   pip install huggingface_hub")
    
    print("\n2. Run the login command:")
    print("   huggingface-cli login")
    
    print("\n3. Enter your Hugging Face token when prompted")
    print("   (Get your token from: https://huggingface.co/settings/tokens)")
    
    input("\nPress Enter after you've completed the login process...")
    
    if check_huggingface_login():
        print("✅ Login successful!")
        return True
    else:
        print("❌ Login failed. Please try again.")
        return False


def accept_medgemma_terms():
    """Guide user to accept MedGemma model terms."""
    print("\n📋 MedGemma Model Terms Acceptance")
    print("=" * 40)
    
    print("MedGemma requires accepting model terms before use.")
    print("\n1. Visit the MedGemma model page:")
    print("   https://huggingface.co/google/medgemma-2b")
    
    print("\n2. Click the 'Accept' button to agree to the terms")
    
    print("\n3. Alternative models you can use:")
    print("   - google/medgemma-2b (2B parameters)")
    print("   - google/medgemma-7b (7B parameters)")
    print("   - google/medgemma-2b-it (instruction-tuned)")
    
    input("\nPress Enter after you've accepted the model terms...")
    
    return True


def test_medgemma_access():
    """Test if MedGemma model can be accessed."""
    print("\n🧪 Testing MedGemma Access")
    print("=" * 40)
    
    try:
        from transformers import AutoTokenizer
        
        print("Testing access to MedGemma model...")
        tokenizer = AutoTokenizer.from_pretrained(
            "google/medgemma-2b",
            trust_remote_code=True
        )
        print("✅ Successfully loaded MedGemma tokenizer!")
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "unauthorized" in error_msg.lower():
            print("❌ Authentication Error: You need to accept the model terms")
            print("   Visit: https://huggingface.co/google/medgemma-2b")
            return False
        else:
            print(f"❌ Error: {error_msg}")
            return False


def install_dependencies():
    """Install required dependencies for MedGemma."""
    print("\n📦 Installing MedGemma Dependencies")
    print("=" * 40)
    
    # Core dependencies that should work on all platforms
    core_dependencies = [
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "huggingface_hub>=0.19.0"
    ]
    
    for dep in core_dependencies:
        print(f"Installing {dep}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], check=True)
            print(f"✅ {dep} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {dep}")
            return False
    
    # Try to install sentencepiece (optional for some models)
    print("Installing sentencepiece>=0.1.99...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "sentencepiece>=0.1.99"], check=True)
        print("✅ sentencepiece>=0.1.99 installed successfully")
    except subprocess.CalledProcessError:
        print("⚠️  Failed to install sentencepiece (this may be needed for some MedGemma models)")
        print("   You can try installing it manually later if needed")
        print("   For Windows, you might need to install Visual Studio Build Tools")
    
    return True


def create_medgemma_config():
    """Create configuration file for MedGemma."""
    print("\n⚙️  Creating MedGemma Configuration")
    print("=" * 40)
    
    config_content = """# MedGemma Configuration
# This file contains settings for MedGemma document parser

# Model settings
MEDGEMMA_MODEL = "google/medgemma-2b"
CACHE_DIR = "./medgemma_cache"

# Authentication
# Set your Hugging Face token here or use environment variable HF_TOKEN
# HF_TOKEN = "your_token_here"

# Performance settings
USE_CUDA = true
LOAD_IN_8BIT = false
LOAD_IN_4BIT = false

# Processing settings
MAX_LENGTH = 512
TEMPERATURE = 0.1
TOP_P = 0.9
"""
    
    config_file = Path("medgemma_config.py")
    config_file.write_text(config_content)
    print(f"✅ Created {config_file}")
    
    return True


def main():
    """Main setup function."""
    print("🏥 MedGemma Setup Script")
    print("=" * 50)
    print("This script will help you set up MedGemma for medical document parsing.")
    
    # Step 1: Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        return False
    
    # Step 2: Check login status
    if not check_huggingface_login():
        if not login_to_huggingface():
            print("❌ Login failed. Cannot proceed.")
            return False
    
    # Step 3: Accept model terms
    accept_medgemma_terms()
    
    # Step 4: Test access
    if not test_medgemma_access():
        print("❌ Cannot access MedGemma model. Please check your authentication.")
        return False
    
    # Step 5: Create configuration
    create_medgemma_config()
    
    print("\n🎉 MedGemma Setup Complete!")
    print("=" * 50)
    print("You can now use MedGemma for medical document parsing:")
    print("\n1. Run the MedGemma parser:")
    print("   python medgemma_document_parser.py")
    
    print("\n2. Or use it in your code:")
    print("   from medgemma_document_parser import MedGemmaDocumentParser")
    print("   parser = MedGemmaDocumentParser()")
    
    print("\n3. For different MedGemma models:")
    print("   parser = MedGemmaDocumentParser('google/medgemma-7b')")
    print("   parser = MedGemmaDocumentParser('google/medgemma-2b-it')")
    
    return True


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 