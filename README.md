# 🤗 Hugging Face Model Runner

A comprehensive Python project for running Hugging Face models locally on your machine. This project provides easy-to-use interfaces for text generation, text classification, image classification, and translation tasks.

## 🚀 Features

- **Text Generation**: Generate creative and coherent text using models like GPT-2
- **Text Classification**: Analyze sentiment and classify text using BERT-based models
- **Image Classification**: Classify images using Vision Transformer (ViT) and ResNet models
- **Translation**: Translate text between multiple languages
- **Easy Configuration**: Simple configuration system for different model types
- **GPU Support**: Automatic GPU detection and optimization
- **Interactive Mode**: Interactive sessions for real-time model testing
- **Batch Processing**: Process multiple inputs efficiently

## 📋 Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional, for faster inference)

## 🛠️ Installation

1. **Clone or download this project:**
   ```bash
   git clone <repository-url>
   cd huggingface
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **For GPU support (optional):**
   - If you have a CUDA-capable GPU, uncomment the CUDA line in `requirements.txt`
   - Or install PyTorch with CUDA support manually:
     ```bash
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
     ```

## 🎯 Quick Start

### Basic Usage

Run the main demo to see all model types in action:

```bash
python main.py
```

### Text Generation

```bash
# Run text generation examples
python examples/text_generation.py

# Interactive text generation
python examples/text_generation.py interactive

# Generate text from command line
python main.py text-gen "The future of AI is"
```

### Text Classification

```bash
# Run classification examples
python examples/text_classification.py

# Interactive classification
python examples/text_classification.py interactive

# Classify text from command line
python main.py classify "I love this movie!"
```

### Image Classification

```bash
# Run image classification examples
python examples/image_classification.py

# Interactive image classification
python examples/image_classification.py interactive

# Compare different models
python examples/image_classification.py compare
```

### Translation

```bash
# Run translation examples
python examples/translation.py

# Interactive translation
python examples/translation.py interactive

# Multi-language translation
python examples/translation.py multi

# Translate from command line
python main.py translate "Hello, how are you?"
```

## ⚙️ Configuration

The project uses a configuration system in `config.py` that allows you to:

- Set default models for each task
- Configure model parameters (temperature, max_length, etc.)
- Control device usage (CPU/GPU)
- Set cache directory for downloaded models
- Enable quantization for memory optimization

### Environment Variables

You can customize behavior using environment variables:

```bash
# Use CPU instead of GPU
export USE_CUDA=false

# Set custom cache directory
export HF_CACHE_DIR=/path/to/cache

# Set batch size
export BATCH_SIZE=4

# Enable 8-bit quantization (saves memory)
export LOAD_IN_8BIT=true
```

## 📁 Project Structure

```
huggingface/
├── main.py                 # Main model runner class
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore           # Git ignore rules
├── setup.py             # Package setup (optional)
└── examples/            # Example scripts
    ├── text_generation.py
    ├── text_classification.py
    ├── image_classification.py
    └── translation.py
```

## 🔧 Customization

### Using Different Models

You can easily switch to different models by modifying the `DEFAULT_MODELS` in `config.py`:

```python
DEFAULT_MODELS = {
    "text_generation": "microsoft/DialoGPT-medium",
    "text_classification": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "translation": "Helsinki-NLP/opus-mt-en-es",
    "image_classification": "microsoft/resnet-50"
}
```

### Adding New Model Types

To add support for new model types:

1. Add the model configuration to `config.py`
2. Create a new method in `HuggingFaceModelRunner` class
3. Add an example script in the `examples/` directory

## 🎮 Interactive Usage

The project includes interactive modes for real-time testing:

```python
from main import HuggingFaceModelRunner

# Initialize the runner
runner = HuggingFaceModelRunner()

# Load and use models
runner.load_text_generation_model()
result = runner.generate_text("Once upon a time")

# Classify text
classification = runner.classify_text("I love this product!")

# Translate text
translation = runner.translate_text("Hello world")

# Classify image
image_result = runner.classify_image("path/to/image.jpg")
```

## 🚀 Performance Tips

1. **GPU Usage**: Models run much faster on GPU. Ensure CUDA is properly installed.
2. **Model Caching**: Models are automatically cached in the `models_cache` directory.
3. **Quantization**: Use 8-bit or 4-bit quantization for large models to save memory.
4. **Batch Processing**: Process multiple inputs together for better efficiency.

## 🔍 Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use smaller models
   - Enable quantization (`LOAD_IN_8BIT=true`)
   - Use CPU instead of GPU

2. **Model Download Issues**:
   - Check internet connection
   - Clear cache directory
   - Use a different model

3. **CUDA Issues**:
   - Verify CUDA installation
   - Check PyTorch CUDA version
   - Set `USE_CUDA=false` to use CPU

### Getting Help

- Check the example scripts for usage patterns
- Review the configuration options in `config.py`
- Ensure all dependencies are properly installed

## 📚 Model Information

### Supported Model Types

- **Text Generation**: GPT-2, DialoGPT, and other causal language models
- **Text Classification**: BERT, RoBERTa, and DistilBERT variants
- **Image Classification**: ViT, ResNet, and DeiT models
- **Translation**: MarianMT and OPUS-MT models

### Model Sources

All models are downloaded from the Hugging Face Hub. You can browse available models at:
- [Hugging Face Models](https://huggingface.co/models)
- [Model Hub Documentation](https://huggingface.co/docs/hub/index)

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding new model types
- Improving examples
- Enhancing documentation
- Reporting issues

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the amazing transformers library
- [PyTorch](https://pytorch.org/) for the deep learning framework
- The open-source community for the models and tools

---

**Happy modeling! 🎉** 