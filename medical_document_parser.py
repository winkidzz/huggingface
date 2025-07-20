#!/usr/bin/env python3
"""
Medical Document Parser
A Python program that uses Hugging Face models to parse medical documents and extract structured data.
This version uses publicly available models that we already have cached.
"""

import os
import sys
import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd


@dataclass
class MedicalEntity:
    """Data class for medical entities extracted from documents."""
    entity_type: str
    value: str
    confidence: float
    context: Optional[str] = None


@dataclass
class DocumentSection:
    """Data class for document sections."""
    section_name: str
    content: str
    entities: List[MedicalEntity]
    summary: Optional[str] = None


@dataclass
class ParsedDocument:
    """Data class for parsed document results."""
    document_id: str
    document_type: str
    patient_info: Dict[str, Any]
    medical_entities: List[MedicalEntity]
    sections: List[DocumentSection]
    summary: str
    extraction_date: str
    confidence_score: float


class MedicalDocumentParser:
    """Main class for parsing medical documents using Hugging Face models."""
    
    def __init__(self, model_name: str = "gpt2", cache_dir: str = "./models_cache"):
        """Initialize the medical document parser."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"🏥 Initializing Medical Document Parser")
        print(f"📁 Cache directory: {self.cache_dir}")
        print(f"🖥️  Device: {self.device}")
        print(f"🤖 Model: {model_name}")
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"📥 Loading model: {self.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            print("✅ Model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """Generate response using the loaded model."""
        if self.model is None:
            self.load_model()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the original prompt from the response
            response = response.replace(prompt, "").strip()
            
            return response
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return ""
    
    def extract_patient_info_rule_based(self, document_text: str) -> Dict[str, Any]:
        """Extract patient information using rule-based parsing."""
        patient_info = {}
        
        # Extract patient name
        name_match = re.search(r'PATIENT:\s*([^\n]+)', document_text, re.IGNORECASE)
        if name_match:
            patient_info['patient_name'] = name_match.group(1).strip()
        
        # Extract DOB
        dob_match = re.search(r'DOB:\s*([^\n]+)', document_text, re.IGNORECASE)
        if dob_match:
            patient_info['date_of_birth'] = dob_match.group(1).strip()
        
        # Extract MRN
        mrn_match = re.search(r'MRN:\s*([^\n]+)', document_text, re.IGNORECASE)
        if mrn_match:
            patient_info['medical_record_number'] = mrn_match.group(1).strip()
        
        # Extract chief complaint
        cc_match = re.search(r'CHIEF COMPLAINT:\s*([^\n]+)', document_text, re.IGNORECASE)
        if cc_match:
            patient_info['chief_complaint'] = cc_match.group(1).strip()
        
        # Extract diagnosis
        diagnosis_match = re.search(r'DIAGNOSIS:\s*([^\n]+)', document_text, re.IGNORECASE)
        if diagnosis_match:
            patient_info['primary_diagnosis'] = diagnosis_match.group(1).strip()
        
        # Extract medications
        med_match = re.search(r'MEDICATIONS:\s*([^\n]+)', document_text, re.IGNORECASE)
        if med_match:
            patient_info['medications'] = med_match.group(1).strip()
        
        # Extract allergies
        allergy_match = re.search(r'ALLERGIES:\s*([^\n]+)', document_text, re.IGNORECASE)
        if allergy_match:
            patient_info['allergies'] = allergy_match.group(1).strip()
        
        return patient_info
    
    def extract_medical_entities_rule_based(self, document_text: str) -> List[MedicalEntity]:
        """Extract medical entities using rule-based parsing."""
        entities = []
        
        # Extract diagnoses
        diagnosis_patterns = [
            r'DIAGNOSIS:\s*([^\n]+)',
            r'ASSESSMENT:\s*([^\n]+)',
            r'IMPRESSION:\s*([^\n]+)'
        ]
        
        for pattern in diagnosis_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity_type="DIAGNOSIS",
                    value=match.group(1).strip(),
                    confidence=0.9,
                    context=match.group(0)
                ))
        
        # Extract medications
        med_patterns = [
            r'MEDICATIONS:\s*([^\n]+)',
            r'CURRENT MEDICATIONS:\s*([^\n]+)'
        ]
        
        for pattern in med_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity_type="MEDICATION",
                    value=match.group(1).strip(),
                    confidence=0.9,
                    context=match.group(0)
                ))
        
        # Extract lab results
        lab_patterns = [
            r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]+)\s*\(Reference:\s*([^)]+)\)',
            r'([A-Za-z\s]+):\s*([0-9.]+)\s*([A-Za-z/%]+)'
        ]
        
        for pattern in lab_patterns:
            matches = re.finditer(pattern, document_text)
            for match in matches:
                if len(match.groups()) >= 3:
                    entities.append(MedicalEntity(
                        entity_type="LAB_RESULT",
                        value=f"{match.group(1).strip()}: {match.group(2)} {match.group(3)}",
                        confidence=0.8,
                        context=match.group(0)
                    ))
        
        # Extract vital signs
        vital_patterns = [
            r'BP\s*([0-9]+/[0-9]+)',
            r'HR\s*([0-9]+)',
            r'T\s*([0-9.]+)F',
            r'O2\s*sat\s*([0-9]+)%'
        ]
        
        vital_names = ["Blood Pressure", "Heart Rate", "Temperature", "Oxygen Saturation"]
        
        for i, pattern in enumerate(vital_patterns):
            matches = re.finditer(pattern, document_text, re.IGNORECASE)
            for match in matches:
                entities.append(MedicalEntity(
                    entity_type="VITAL_SIGN",
                    value=f"{vital_names[i]}: {match.group(1)}",
                    confidence=0.9,
                    context=match.group(0)
                ))
        
        return entities
    
    def extract_document_sections(self, document_text: str) -> List[DocumentSection]:
        """Extract and parse document sections."""
        # Common medical document sections
        section_patterns = [
            r'(CHIEF COMPLAINT|PRESENT ILLNESS|HISTORY OF PRESENT ILLNESS)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(PAST MEDICAL HISTORY|MEDICAL HISTORY)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(MEDICATIONS|CURRENT MEDICATIONS)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(ALLERGIES|DRUG ALLERGIES)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(PHYSICAL EXAMINATION|EXAMINATION)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(LABORATORY RESULTS|LAB RESULTS)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(ASSESSMENT|DIAGNOSIS)[:\s]*(.*?)(?=\n[A-Z]|$)',
            r'(PLAN|TREATMENT PLAN)[:\s]*(.*?)(?=\n[A-Z]|$)'
        ]
        
        sections = []
        
        for pattern in section_patterns:
            matches = re.finditer(pattern, document_text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_name = match.group(1).strip()
                content = match.group(2).strip()
                
                if content and len(content) > 10:  # Only add non-empty sections
                    # Extract entities from this section
                    entities = self.extract_medical_entities_rule_based(content)
                    
                    # Generate summary for this section
                    summary = f"Section contains {len(entities)} medical entities"
                    
                    section = DocumentSection(
                        section_name=section_name,
                        content=content,
                        entities=entities,
                        summary=summary
                    )
                    sections.append(section)
        
        return sections
    
    def generate_document_summary(self, document_text: str) -> str:
        """Generate a summary of the medical document."""
        # Extract key information for summary
        patient_name = ""
        chief_complaint = ""
        diagnosis = ""
        
        name_match = re.search(r'PATIENT:\s*([^\n]+)', document_text, re.IGNORECASE)
        if name_match:
            patient_name = name_match.group(1).strip()
        
        cc_match = re.search(r'CHIEF COMPLAINT:\s*([^\n]+)', document_text, re.IGNORECASE)
        if cc_match:
            chief_complaint = cc_match.group(1).strip()
        
        diag_match = re.search(r'DIAGNOSIS:\s*([^\n]+)', document_text, re.IGNORECASE)
        if diag_match:
            diagnosis = diag_match.group(1).strip()
        
        summary = f"Medical document for {patient_name}. "
        if chief_complaint:
            summary += f"Chief complaint: {chief_complaint}. "
        if diagnosis:
            summary += f"Diagnosis: {diagnosis}. "
        
        summary += f"Document contains {len(self.extract_medical_entities_rule_based(document_text))} medical entities."
        
        return summary
    
    def parse_document(self, document_text: str, document_id: str = None, document_type: str = "medical_note") -> ParsedDocument:
        """Parse a medical document and extract structured data."""
        if document_id is None:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📄 Parsing document: {document_id}")
        
        # Extract patient information
        print("🔍 Extracting patient information...")
        patient_info = self.extract_patient_info_rule_based(document_text)
        
        # Extract medical entities
        print("🏥 Extracting medical entities...")
        medical_entities = self.extract_medical_entities_rule_based(document_text)
        
        # Extract document sections
        print("📋 Extracting document sections...")
        sections = self.extract_document_sections(document_text)
        
        # Generate document summary
        print("📝 Generating document summary...")
        summary = self.generate_document_summary(document_text)
        
        # Calculate overall confidence score
        if medical_entities:
            confidence_score = sum(entity.confidence for entity in medical_entities) / len(medical_entities)
        else:
            confidence_score = 0.0
        
        parsed_doc = ParsedDocument(
            document_id=document_id,
            document_type=document_type,
            patient_info=patient_info,
            medical_entities=medical_entities,
            sections=sections,
            summary=summary,
            extraction_date=datetime.now().isoformat(),
            confidence_score=confidence_score
        )
        
        print(f"✅ Document parsing completed (Confidence: {confidence_score:.2f})")
        return parsed_doc
    
    def save_results(self, parsed_doc: ParsedDocument, output_dir: str = "./parsed_documents"):
        """Save parsing results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSON
        json_file = os.path.join(output_dir, f"{parsed_doc.document_id}.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(parsed_doc), f, indent=2, default=str)
        
        # Save entities as CSV
        if parsed_doc.medical_entities:
            csv_file = os.path.join(output_dir, f"{parsed_doc.document_id}_entities.csv")
            entities_data = []
            for entity in parsed_doc.medical_entities:
                entities_data.append({
                    'entity_type': entity.entity_type,
                    'value': entity.value,
                    'confidence': entity.confidence,
                    'context': entity.context
                })
            
            df = pd.DataFrame(entities_data)
            df.to_csv(csv_file, index=False)
        
        # Save sections as text
        txt_file = os.path.join(output_dir, f"{parsed_doc.document_id}_sections.txt")
        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write(f"Document ID: {parsed_doc.document_id}\n")
            f.write(f"Document Type: {parsed_doc.document_type}\n")
            f.write(f"Extraction Date: {parsed_doc.extraction_date}\n")
            f.write(f"Confidence Score: {parsed_doc.confidence_score:.2f}\n\n")
            
            f.write("SUMMARY:\n")
            f.write(parsed_doc.summary)
            f.write("\n\n")
            
            f.write("PATIENT INFORMATION:\n")
            f.write(json.dumps(parsed_doc.patient_info, indent=2))
            f.write("\n\n")
            
            f.write("SECTIONS:\n")
            for section in parsed_doc.sections:
                f.write(f"\n{section.section_name.upper()}:\n")
                f.write(f"Summary: {section.summary}\n")
                f.write(f"Content: {section.content[:200]}...\n")
                f.write(f"Entities: {len(section.entities)}\n")
        
        print(f"💾 Results saved to {output_dir}/")
        return json_file, csv_file if parsed_doc.medical_entities else None, txt_file
    
    def batch_parse_documents(self, document_files: List[str], output_dir: str = "./parsed_documents") -> List[ParsedDocument]:
        """Parse multiple documents in batch."""
        results = []
        
        for i, file_path in enumerate(document_files, 1):
            print(f"\n📄 Processing document {i}/{len(document_files)}: {file_path}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    document_text = f.read()
                
                document_id = os.path.splitext(os.path.basename(file_path))[0]
                parsed_doc = self.parse_document(document_text, document_id)
                
                # Save results
                self.save_results(parsed_doc, output_dir)
                results.append(parsed_doc)
                
            except Exception as e:
                print(f"❌ Error processing {file_path}: {e}")
        
        return results


def main():
    """Main function to demonstrate medical document parsing."""
    print("🏥 Medical Document Parser")
    print("=" * 50)
    
    # Initialize parser
    parser = MedicalDocumentParser()
    
    # Sample medical document
    sample_document = """
    PATIENT: John Smith
    DOB: 01/15/1980
    MRN: 123456789
    DATE: 2024-01-15
    
    CHIEF COMPLAINT: Chest pain and shortness of breath for 2 days
    
    HISTORY OF PRESENT ILLNESS: Mr. Smith is a 44-year-old male who presents with chest pain and shortness of breath for the past 2 days. The pain is described as pressure-like, radiating to the left arm, and is worse with exertion. He also reports associated diaphoresis and nausea.
    
    PAST MEDICAL HISTORY: Hypertension, Type 2 Diabetes, Hyperlipidemia
    
    MEDICATIONS: Lisinopril 10mg daily, Metformin 500mg twice daily, Atorvastatin 20mg daily
    
    ALLERGIES: Penicillin (rash)
    
    PHYSICAL EXAMINATION: Vital signs: BP 160/95, HR 95, RR 22, T 98.6F, O2 sat 94% on RA
    General: Alert and oriented, in mild distress
    Cardiovascular: Regular rate and rhythm, S1/S2 normal, no murmurs
    Respiratory: Clear to auscultation bilaterally
    Abdomen: Soft, non-tender, non-distended
    
    LABORATORY RESULTS: Troponin I: 0.15 ng/mL (Reference: 0.00-0.04) - HIGH, CK-MB: 25 ng/mL, ECG shows ST elevation in leads II, III, aVF
    
    ASSESSMENT: Acute inferior wall myocardial infarction
    
    PLAN: 
    1. Admit to cardiac ICU
    2. Start aspirin 325mg, clopidogrel 600mg loading dose
    3. Consult cardiology for emergent cardiac catheterization
    4. Monitor cardiac enzymes and ECG
    5. Start heparin drip
    """
    
    # Parse the document
    print("🔍 Parsing sample medical document...")
    parsed_doc = parser.parse_document(sample_document, "sample_medical_note")
    
    # Save results
    parser.save_results(parsed_doc)
    
    # Display results
    print("\n📊 PARSING RESULTS:")
    print("=" * 50)
    print(f"Document ID: {parsed_doc.document_id}")
    print(f"Confidence Score: {parsed_doc.confidence_score:.2f}")
    print(f"Medical Entities Found: {len(parsed_doc.medical_entities)}")
    print(f"Document Sections: {len(parsed_doc.sections)}")
    
    print("\n🏥 MEDICAL ENTITIES:")
    for entity in parsed_doc.medical_entities[:10]:  # Show first 10
        print(f"  {entity.entity_type}: {entity.value} (confidence: {entity.confidence:.2f})")
    
    print("\n📋 DOCUMENT SECTIONS:")
    for section in parsed_doc.sections:
        print(f"  {section.section_name}: {len(section.entities)} entities")
    
    print("\n📝 SUMMARY:")
    print(parsed_doc.summary)
    
    print("\n✅ Document parsing completed successfully!")


if __name__ == "__main__":
    main() 