# 🏥 MedGemma Document Parser

A Python program that uses Google's MedGemma model to parse medical documents and extract structured data. This tool is specifically designed for processing medical notes, lab reports, radiology reports, and other healthcare documents.

## 🚀 Features

- **Medical Document Parsing**: Extract structured data from various medical document types
- **Entity Extraction**: Identify medical entities like diagnoses, medications, lab results, and procedures
- **Patient Information Extraction**: Automatically extract patient demographics and medical history
- **Document Sectioning**: Parse documents into logical sections (chief complaint, history, assessment, etc.)
- **Multi-format Output**: Save results as JSON, CSV, and text files
- **Batch Processing**: Process multiple documents efficiently
- **Interactive Mode**: Real-time document parsing and analysis
- **Confidence Scoring**: Provide confidence scores for extracted information

## 📋 Requirements

- Python 3.8 or higher
- PyTorch 2.0 or higher
- CUDA-compatible GPU (optional, for faster processing)

## 🛠️ Installation

1. **Install dependencies:**
   ```bash
   pip install -r medgemma_requirements.txt
   ```

2. **For GPU support (optional):**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## 🎯 Quick Start

### Basic Usage

```python
from medgemma_document_parser import MedGemmaDocumentParser

# Initialize parser
parser = MedGemmaDocumentParser()

# Parse a medical document
document_text = """
PATIENT: John Smith
CHIEF COMPLAINT: Chest pain
DIAGNOSIS: Angina
MEDICATIONS: Nitroglycerin
"""

parsed_doc = parser.parse_document(document_text, "john_smith_note")

# Access results
print(f"Confidence: {parsed_doc.confidence_score}")
print(f"Entities: {len(parsed_doc.medical_entities)}")
print(f"Summary: {parsed_doc.summary}")
```

### Command Line Usage

```bash
# Run the main parser with sample document
python medgemma_document_parser.py

# Run specific examples
python medgemma_examples.py medical_note
python medgemma_examples.py lab_report
python medgemma_examples.py radiology

# Interactive mode
python medgemma_examples.py interactive

# Batch processing
python medgemma_examples.py batch
```

## 📊 Supported Document Types

### 1. Medical Notes
- Chief complaints
- History of present illness
- Past medical history
- Physical examination
- Assessment and plan

### 2. Laboratory Reports
- Complete blood count (CBC)
- Comprehensive metabolic panel
- Lipid panel
- Thyroid function tests
- Other lab results

### 3. Radiology Reports
- Imaging findings
- Clinical impressions
- Recommendations
- Technical details

### 4. Medication Lists
- Current medications
- Dosages and frequencies
- Allergies and contraindications

## 🔧 Configuration

### Model Settings

```python
# Use different MedGemma model
parser = MedGemmaDocumentParser(model_name="google/medgemma-2b")

# Set custom cache directory
parser = MedGemmaDocumentParser(cache_dir="./my_cache")
```

### Environment Variables

```bash
# Use CPU instead of GPU
export USE_CUDA=false

# Set custom cache directory
export HF_CACHE_DIR=/path/to/cache

# Enable memory optimization
export LOAD_IN_8BIT=true
```

## 📁 Output Formats

### JSON Output
```json
{
  "document_id": "john_smith_note",
  "document_type": "medical_note",
  "patient_info": {
    "patient_name": "John Smith",
    "primary_diagnosis": "Angina"
  },
  "medical_entities": [
    {
      "entity_type": "DIAGNOSIS",
      "value": "Angina",
      "confidence": 0.95
    }
  ],
  "summary": "Patient presents with chest pain...",
  "confidence_score": 0.92
}
```

### CSV Output
```csv
entity_type,value,confidence,context
DIAGNOSIS,Angina,0.95,"Patient diagnosed with angina"
MEDICATION,Nitroglycerin,0.88,"Prescribed nitroglycerin for chest pain"
```

## 🏥 Medical Entity Types

The parser extracts various types of medical entities:

- **DIAGNOSIS**: Medical conditions and diagnoses
- **MEDICATION**: Drugs and dosages
- **PROCEDURE**: Medical procedures and interventions
- **SYMPTOM**: Patient symptoms and complaints
- **LAB_RESULT**: Laboratory test results
- **VITAL_SIGN**: Blood pressure, heart rate, temperature, etc.
- **ALLERGY**: Drug allergies and reactions
- **FINDING**: Clinical findings and observations

## 📈 Performance Tips

1. **GPU Usage**: MedGemma runs much faster on GPU. Ensure CUDA is properly installed.
2. **Model Caching**: Models are automatically cached locally for faster subsequent runs.
3. **Batch Processing**: Process multiple documents together for better efficiency.
4. **Memory Optimization**: Use 8-bit quantization for large documents.

## 🔍 Example Use Cases

### Clinical Documentation
```python
# Parse clinical notes for quality improvement
clinical_notes = ["note1.txt", "note2.txt", "note3.txt"]
results = parser.batch_parse_documents(clinical_notes)

# Extract all diagnoses
all_diagnoses = []
for doc in results:
    diagnoses = [e for e in doc.medical_entities if e.entity_type == "DIAGNOSIS"]
    all_diagnoses.extend(diagnoses)
```

### Research Data Extraction
```python
# Extract lab results for research analysis
lab_reports = ["lab1.txt", "lab2.txt"]
results = parser.batch_parse_documents(lab_reports)

# Create dataset of lab values
lab_data = []
for doc in results:
    lab_entities = [e for e in doc.medical_entities if "LAB" in e.entity_type]
    lab_data.extend(lab_entities)
```

### Quality Assurance
```python
# Check for missing critical information
def check_completeness(parsed_doc):
    required_sections = ["CHIEF_COMPLAINT", "ASSESSMENT", "PLAN"]
    missing_sections = []
    
    for section in required_sections:
        if not any(s.section_name == section for s in parsed_doc.sections):
            missing_sections.append(section)
    
    return missing_sections
```

## 🚨 Important Notes

### Privacy and Security
- This tool processes medical documents that may contain sensitive information
- Ensure compliance with HIPAA and other privacy regulations
- Consider data anonymization for research purposes
- Use secure environments for processing sensitive documents

### Model Limitations
- MedGemma is trained on medical data but may not be 100% accurate
- Always review extracted information for clinical use
- The model may not recognize all medical terminology or abbreviations
- Results should be validated by healthcare professionals

### Performance Considerations
- Large documents may require significant memory
- Processing time depends on document length and hardware
- Consider chunking very long documents for better performance

## 🔧 Troubleshooting

### Common Issues

1. **Out of Memory**: 
   - Use smaller model variants
   - Enable quantization (`LOAD_IN_8BIT=true`)
   - Process documents in smaller chunks

2. **Model Download Issues**:
   - Check internet connection
   - Clear cache directory
   - Verify Hugging Face Hub access

3. **Low Confidence Scores**:
   - Check document quality and formatting
   - Ensure medical terminology is clear
   - Consider preprocessing documents

### Getting Help

- Check the example scripts for usage patterns
- Review the configuration options
- Ensure all dependencies are properly installed
- Check MedGemma documentation for model-specific issues

## 📚 References

- [MedGemma Paper](https://arxiv.org/abs/2401.08417)
- [Hugging Face MedGemma](https://huggingface.co/google/medgemma-2b)
- [Transformers Documentation](https://huggingface.co/docs/transformers/)

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding support for new document types
- Improving entity extraction accuracy
- Enhancing the output formats
- Adding new medical entity types
- Reporting issues and bugs

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- [Google Research](https://research.google/) for MedGemma
- [Hugging Face](https://huggingface.co/) for the transformers library
- The medical AI research community

---

**⚠️ Disclaimer**: This tool is for educational and research purposes. Always consult healthcare professionals for medical decisions. The extracted information should be validated before clinical use. 