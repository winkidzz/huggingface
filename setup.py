#!/usr/bin/env python3
"""
Setup script for Hugging Face Model Runner
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="huggingface-model-runner",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A comprehensive Python project for running Hugging Face models locally",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/huggingface-model-runner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "gpu": [
            "torch>=2.0.0+cu118",
            "torchvision>=0.15.0+cu118",
            "torchaudio>=2.0.0+cu118",
        ],
    },
    entry_points={
        "console_scripts": [
            "hf-runner=main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="huggingface transformers machine-learning ai nlp computer-vision",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/huggingface-model-runner/issues",
        "Source": "https://github.com/yourusername/huggingface-model-runner",
        "Documentation": "https://github.com/yourusername/huggingface-model-runner#readme",
    },
) 