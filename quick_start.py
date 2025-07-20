#!/usr/bin/env python3
"""
Quick Start Script for Hugging Face Model Runner
This script helps you get started quickly with the project.
"""

import os
import sys
import subprocess
import platform


def print_banner():
    """Print a welcome banner."""
    print("🤗" + "=" * 60)
    print("    Welcome to Hugging Face Model Runner!")
    print("=" * 62)
    print()


def check_python_version():
    """Check if Python version is compatible."""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("   This project requires Python 3.8 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True


def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    required_packages = [
        "torch",
        "transformers",
        "PIL",
        "numpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == "PIL":
                import PIL
                print(f"✅ {package} (Pillow) is installed")
            else:
                __import__(package)
                print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    return True


def install_dependencies():
    """Install dependencies if needed."""
    print("\n📦 Installing dependencies...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print(f"❌ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False


def run_setup_test():
    """Run the setup test."""
    print("\n🧪 Running setup test...")
    
    try:
        result = subprocess.run([
            sys.executable, "test_setup.py"
        ], capture_output=True, text=True)
        
        print(result.stdout)
        
        if result.returncode == 0:
            print("✅ Setup test passed!")
            return True
        else:
            print("❌ Setup test failed")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def show_next_steps():
    """Show next steps for the user."""
    print("\n🎯 Next Steps:")
    print("=" * 40)
    print("1. 🚀 Run the main demo:")
    print("   python main.py")
    print()
    print("2. 📝 Try text generation:")
    print("   python examples/text_generation.py")
    print()
    print("3. 🏷️  Try text classification:")
    print("   python examples/text_classification.py")
    print()
    print("4. 🖼️  Try image classification:")
    print("   python examples/image_classification.py")
    print()
    print("5. 🌐 Try translation:")
    print("   python examples/translation.py")
    print()
    print("6. 📖 Read the documentation:")
    print("   Check README.md for detailed instructions")
    print()
    print("7. ⚙️  Customize configuration:")
    print("   Edit config.py to change models and settings")
    print()


def show_troubleshooting():
    """Show troubleshooting tips."""
    print("\n🔧 Troubleshooting Tips:")
    print("=" * 40)
    print("• If models download slowly, check your internet connection")
    print("• If you get memory errors, try using smaller models")
    print("• For GPU issues, set USE_CUDA=false environment variable")
    print("• Clear the models_cache directory if downloads fail")
    print("• Check the README.md for more detailed help")
    print()


def main():
    """Main quick start function."""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Please upgrade Python to version 3.8 or higher")
        return False
    
    # Check dependencies
    if not check_dependencies():
        print("\n📦 Installing missing dependencies...")
        if not install_dependencies():
            print("\n❌ Failed to install dependencies")
            print("   Please run: pip install -r requirements.txt")
            return False
    
    # Run setup test
    if not run_setup_test():
        print("\n❌ Setup test failed")
        show_troubleshooting()
        return False
    
    # Show next steps
    show_next_steps()
    
    print("🎉 Setup completed successfully!")
    print("   You're ready to start using Hugging Face models!")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. You can run this script again later.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1) 