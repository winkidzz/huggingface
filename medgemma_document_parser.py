#!/usr/bin/env python3
"""
MedGemma Document Parser
A Python program that uses MedGemma to parse medical documents and extract structured data.
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
from PIL import Image
import requests
from io import BytesIO
import pandas as pd


@dataclass
class MedicalEntity:
    """Data class for medical entities extracted from documents."""
    entity_type: str
    value: str
    confidence: float
    start_position: Optional[int] = None
    end_position: Optional[int] = None
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


class MedGemmaDocumentParser:
    """Main class for parsing medical documents using MedGemma."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium", cache_dir: str = "./medgemma_cache"):
        """Initialize the MedGemma document parser."""
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None
        
        # Create cache directory
        os.makedirs(self.cache_dir, exist_ok=True)
        
        print(f"🚀 Initializing MedGemma Document Parser")
        print(f"📁 Cache directory: {self.cache_dir}")
        print(f"🖥️  Device: {self.device}")
        print(f"🤖 Model: {model_name}")
    
    def load_model(self):
        """Load the MedGemma model and tokenizer."""
        print(f"📥 Loading MedGemma model: {self.model_name}")
        
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
            
            print("✅ MedGemma model loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading MedGemma model: {e}")
            raise
    
    def generate_response(self, prompt: str, max_length: int = 512, temperature: float = 0.1) -> str:
        """Generate response using MedGemma."""
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
    
    def extract_patient_info(self, document_text: str) -> Dict[str, Any]:
        """Extract patient information from document text."""
        prompt = f"""
        Extract patient information from the following medical document. Return the information in JSON format with these fields:
        - patient_name
        - date_of_birth
        - age
        - gender
        - medical_record_number
        - primary_diagnosis
        - medications
        - allergies
        - vital_signs (blood_pressure, heart_rate, temperature, weight)
        
        Document text:
        {document_text[:2000]}
        
        JSON response:
        """
        
        response = self.generate_response(prompt, max_length=1024, temperature=0.1)
        
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"error": "Could not parse JSON response", "raw_response": response}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON response", "raw_response": response}
    
    def extract_medical_entities(self, document_text: str) -> List[MedicalEntity]:
        """Extract medical entities from document text."""
        prompt = f"""
        Extract medical entities from the following text. For each entity, provide:
        - Entity type (diagnosis, medication, procedure, symptom, lab_result, etc.)
        - Value
        - Confidence (0.0 to 1.0)
        - Context (surrounding text)
        
        Return in this format:
        ENTITY_TYPE: value (confidence: X.XX)
        Context: surrounding text
        
        Text:
        {document_text[:1500]}
        
        Extracted entities:
        """
        
        response = self.generate_response(prompt, max_length=1024, temperature=0.1)
        
        entities = []
        lines = response.split('\n')
        current_entity = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Parse entity line
            entity_match = re.match(r'([A-Z_]+):\s*(.+?)\s*\(confidence:\s*([0-9.]+)\)', line)
            if entity_match:
                entity_type = entity_match.group(1)
                value = entity_match.group(2)
                confidence = float(entity_match.group(3))
                
                current_entity = MedicalEntity(
                    entity_type=entity_type,
                    value=value,
                    confidence=confidence
                )
                entities.append(current_entity)
            
            # Parse context line
            elif line.startswith('Context:') and current_entity:
                context = line.replace('Context:', '').strip()
                current_entity.context = context
        
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
                    entities = self.extract_medical_entities(content)
                    
                    # Generate summary for this section
                    summary_prompt = f"Summarize this medical section in 1-2 sentences: {content[:500]}"
                    summary = self.generate_response(summary_prompt, max_length=200, temperature=0.1)
                    
                    section = DocumentSection(
                        section_name=section_name,
                        content=content,
                        entities=entities,
                        summary=summary
                    )
                    sections.append(section)
        
        return sections
    
    def generate_document_summary(self, document_text: str) -> str:
        """Generate a comprehensive summary of the medical document."""
        prompt = f"""
        Provide a comprehensive summary of this medical document. Include:
        - Main diagnosis or chief complaint
        - Key findings
        - Important medications
        - Treatment plan
        - Critical lab results
        
        Document:
        {document_text[:2000]}
        
        Summary:
        """
        
        return self.generate_response(prompt, max_length=500, temperature=0.1)
    
    def parse_document(self, document_text: str, document_id: str = None, document_type: str = "medical_note") -> ParsedDocument:
        """Parse a medical document and extract structured data."""
        if document_id is None:
            document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"📄 Parsing document: {document_id}")
        
        # Extract patient information
        print("🔍 Extracting patient information...")
        patient_info = self.extract_patient_info(document_text)
        
        # Extract medical entities
        print("🏥 Extracting medical entities...")
        medical_entities = self.extract_medical_entities(document_text)
        
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
    """Main function to demonstrate MedGemma document parsing."""
    print("🏥 MedGemma Document Parser")
    print("=" * 50)
    
    # Initialize parser
    parser = MedGemmaDocumentParser()
    
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
    
    LABORATORY RESULTS: Troponin I: 0.15 ng/mL (elevated), CK-MB: 25 ng/mL, ECG shows ST elevation in leads II, III, aVF
    
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