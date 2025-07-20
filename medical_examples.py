#!/usr/bin/env python3
"""
Medical Document Parser Examples
This script demonstrates how to use the medical document parser for different types of medical documents.
"""

import os
import sys
from medical_document_parser import MedicalDocumentParser


def example_medical_note():
    """Example of parsing a medical note."""
    print("🏥 Medical Note Parsing Example")
    print("=" * 50)
    
    # Sample medical note
    medical_note = """
    PATIENT: Sarah Johnson
    DOB: 03/22/1975
    MRN: 987654321
    DATE: 2024-01-20
    
    CHIEF COMPLAINT: Persistent headache and dizziness for 1 week
    
    HISTORY OF PRESENT ILLNESS: Ms. Johnson is a 48-year-old female who presents with severe headache and dizziness for the past week. The headache is described as throbbing, located in the frontal region, and is associated with photophobia and phonophobia. She also reports nausea and vomiting. The symptoms are worse in the morning and improve throughout the day.
    
    PAST MEDICAL HISTORY: Migraine, Hypertension, Depression
    
    MEDICATIONS: Propranolol 40mg twice daily, Sertraline 50mg daily, Sumatriptan 100mg as needed
    
    ALLERGIES: Sulfa drugs (rash and swelling)
    
    PHYSICAL EXAMINATION: Vital signs: BP 140/90, HR 88, RR 18, T 98.4F, O2 sat 98% on RA
    General: Alert and oriented, appears uncomfortable
    HEENT: Normocephalic, atraumatic, no meningeal signs
    Cardiovascular: Regular rate and rhythm, S1/S2 normal
    Neurological: CN II-XII intact, motor and sensory function normal
    
    LABORATORY RESULTS: CBC: WBC 8.2, Hgb 13.5, Plt 250,000
    Basic metabolic panel: Na 140, K 4.0, Cl 102, CO2 24, BUN 15, Cr 0.9, Glu 95
    
    ASSESSMENT: Migraine with aura, likely triggered by stress and hormonal changes
    
    PLAN: 
    1. Continue current medications
    2. Add Topiramate 25mg daily for migraine prophylaxis
    3. Recommend stress management techniques
    4. Follow up in 2 weeks
    5. Return if symptoms worsen or new neurological symptoms develop
    """
    
    # Initialize parser
    parser = MedicalDocumentParser()
    
    # Parse the document
    parsed_doc = parser.parse_document(medical_note, "sarah_johnson_note")
    
    # Display results
    print(f"📄 Document ID: {parsed_doc.document_id}")
    print(f"🎯 Confidence Score: {parsed_doc.confidence_score:.2f}")
    print(f"🏥 Medical Entities: {len(parsed_doc.medical_entities)}")
    
    print("\n📋 PATIENT INFORMATION:")
    for key, value in parsed_doc.patient_info.items():
        print(f"  {key}: {value}")
    
    print("\n🏥 TOP MEDICAL ENTITIES:")
    for entity in sorted(parsed_doc.medical_entities, key=lambda x: x.confidence, reverse=True)[:5]:
        print(f"  {entity.entity_type}: {entity.value} (confidence: {entity.confidence:.2f})")
    
    print("\n📝 SUMMARY:")
    print(parsed_doc.summary)


def example_lab_report():
    """Example of parsing a laboratory report."""
    print("\n🔬 Laboratory Report Parsing Example")
    print("=" * 50)
    
    lab_report = """
    LABORATORY REPORT
    Patient: Michael Chen
    DOB: 11/08/1982
    MRN: 456789123
    Date: 2024-01-18
    Ordering Physician: Dr. Emily Rodriguez
    
    COMPLETE BLOOD COUNT (CBC):
    White Blood Cell Count: 12.5 K/uL (Reference: 4.5-11.0) - HIGH
    Red Blood Cell Count: 4.8 M/uL (Reference: 4.5-5.9) - NORMAL
    Hemoglobin: 14.2 g/dL (Reference: 13.5-17.5) - NORMAL
    Hematocrit: 42.5% (Reference: 41.0-50.0) - NORMAL
    Platelet Count: 185 K/uL (Reference: 150-450) - NORMAL
    Mean Corpuscular Volume: 88 fL (Reference: 80-100) - NORMAL
    
    COMPREHENSIVE METABOLIC PANEL:
    Glucose: 145 mg/dL (Reference: 70-100) - HIGH
    BUN: 18 mg/dL (Reference: 7-20) - NORMAL
    Creatinine: 1.1 mg/dL (Reference: 0.7-1.3) - NORMAL
    Sodium: 142 mEq/L (Reference: 135-145) - NORMAL
    Potassium: 4.2 mEq/L (Reference: 3.5-5.0) - NORMAL
    Chloride: 104 mEq/L (Reference: 96-106) - NORMAL
    CO2: 24 mEq/L (Reference: 22-28) - NORMAL
    Calcium: 9.8 mg/dL (Reference: 8.5-10.5) - NORMAL
    
    LIPID PANEL:
    Total Cholesterol: 220 mg/dL (Reference: <200) - HIGH
    HDL Cholesterol: 45 mg/dL (Reference: >40) - NORMAL
    LDL Cholesterol: 140 mg/dL (Reference: <100) - HIGH
    Triglycerides: 180 mg/dL (Reference: <150) - HIGH
    
    THYROID FUNCTION TESTS:
    TSH: 2.5 mIU/L (Reference: 0.4-4.0) - NORMAL
    Free T4: 1.2 ng/dL (Reference: 0.8-1.8) - NORMAL
    
    IMPRESSION: 
    - Elevated white blood cell count suggestive of infection or inflammation
    - Elevated glucose consistent with diabetes or prediabetes
    - Elevated lipids indicating hyperlipidemia
    - Recommend follow-up with primary care physician
    """
    
    # Initialize parser
    parser = MedicalDocumentParser()
    
    # Parse the document
    parsed_doc = parser.parse_document(lab_report, "michael_chen_labs")
    
    # Display results
    print(f"📄 Document ID: {parsed_doc.document_id}")
    print(f"🎯 Confidence Score: {parsed_doc.confidence_score:.2f}")
    
    print("\n🔬 LAB RESULTS ENTITIES:")
    lab_entities = [e for e in parsed_doc.medical_entities if "LAB_RESULT" in e.entity_type]
    for entity in lab_entities[:10]:
        print(f"  {entity.entity_type}: {entity.value} (confidence: {entity.confidence:.2f})")
    
    print("\n📝 SUMMARY:")
    print(parsed_doc.summary)


def example_radiology_report():
    """Example of parsing a radiology report."""
    print("\n📷 Radiology Report Parsing Example")
    print("=" * 50)
    
    radiology_report = """
    RADIOLOGY REPORT
    Patient: Lisa Thompson
    DOB: 07/14/1968
    MRN: 789123456
    Date: 2024-01-19
    Study: Chest X-Ray PA and Lateral
    Ordering Physician: Dr. James Wilson
    
    CLINICAL HISTORY: Chest pain, shortness of breath, cough for 3 days
    
    TECHNIQUE: PA and lateral chest radiographs were obtained with the patient in the upright position.
    
    FINDINGS:
    LUNGS: The lungs are clear bilaterally with no evidence of infiltrates, masses, or effusions. The pulmonary vasculature appears normal.
    
    HEART: The cardiac silhouette is normal in size. No evidence of cardiomegaly.
    
    MEDIASTINUM: The mediastinum is normal in width and position. No mediastinal masses or lymphadenopathy identified.
    
    PLEURA: No pleural effusions or pneumothorax identified.
    
    BONES: The visualized osseous structures are unremarkable. No fractures or destructive lesions identified.
    
    SOFT TISSUES: The soft tissues are unremarkable.
    
    IMPRESSION:
    1. Normal chest radiograph
    2. No evidence of pneumonia, pneumothorax, or pleural effusion
    3. No evidence of cardiomegaly or congestive heart failure
    4. Recommend clinical correlation for chest pain symptoms
    
    RECOMMENDATIONS:
    - Consider cardiac evaluation for chest pain
    - Follow up with primary care physician
    - Return for repeat imaging if symptoms persist or worsen
    """
    
    # Initialize parser
    parser = MedicalDocumentParser()
    
    # Parse the document
    parsed_doc = parser.parse_document(radiology_report, "lisa_thompson_chest_xray")
    
    # Display results
    print(f"📄 Document ID: {parsed_doc.document_id}")
    print(f"🎯 Confidence Score: {parsed_doc.confidence_score:.2f}")
    
    print("\n📷 RADIOLOGY FINDINGS:")
    radiology_entities = [e for e in parsed_doc.medical_entities if any(word in e.entity_type.lower() for word in ["finding", "impression", "diagnosis"])]
    for entity in radiology_entities[:8]:
        print(f"  {entity.entity_type}: {entity.value} (confidence: {entity.confidence:.2f})")
    
    print("\n📝 SUMMARY:")
    print(parsed_doc.summary)


def example_batch_processing():
    """Example of batch processing multiple documents."""
    print("\n📚 Batch Processing Example")
    print("=" * 50)
    
    # Create sample documents
    documents = [
        ("patient1_note.txt", """
        PATIENT: Robert Davis
        CHIEF COMPLAINT: Back pain
        DIAGNOSIS: Lumbar strain
        MEDICATIONS: Ibuprofen 800mg
        """),
        ("patient2_note.txt", """
        PATIENT: Maria Garcia
        CHIEF COMPLAINT: Fever and cough
        DIAGNOSIS: Upper respiratory infection
        MEDICATIONS: Amoxicillin 500mg
        """),
        ("patient3_note.txt", """
        PATIENT: David Wilson
        CHIEF COMPLAINT: Chest pain
        DIAGNOSIS: Angina
        MEDICATIONS: Nitroglycerin
        """)
    ]
    
    # Save documents to files
    os.makedirs("./sample_documents", exist_ok=True)
    document_files = []
    
    for filename, content in documents:
        filepath = os.path.join("./sample_documents", filename)
        with open(filepath, 'w') as f:
            f.write(content)
        document_files.append(filepath)
    
    # Initialize parser
    parser = MedicalDocumentParser()
    
    # Process all documents
    print(f"📄 Processing {len(document_files)} documents...")
    results = parser.batch_parse_documents(document_files, "./batch_results")
    
    # Display summary
    print(f"\n✅ Batch processing completed!")
    print(f"📊 Processed {len(results)} documents")
    
    total_entities = sum(len(doc.medical_entities) for doc in results)
    avg_confidence = sum(doc.confidence_score for doc in results) / len(results)
    
    print(f"🏥 Total entities extracted: {total_entities}")
    print(f"🎯 Average confidence: {avg_confidence:.2f}")


def interactive_parsing():
    """Interactive document parsing session."""
    print("\n🎮 Interactive Document Parsing")
    print("=" * 50)
    print("Type 'quit' to exit")
    
    parser = MedicalDocumentParser()
    
    while True:
        try:
            print("\nEnter your medical document text (or 'quit' to exit):")
            print("(You can paste a multi-line document)")
            
            lines = []
            while True:
                line = input()
                if line.lower() == 'quit':
                    return
                if line.strip() == "":
                    break
                lines.append(line)
            
            if not lines:
                continue
            
            document_text = '\n'.join(lines)
            
            # Parse the document
            print("\n🔍 Parsing document...")
            parsed_doc = parser.parse_document(document_text, f"interactive_{len(lines)}")
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"Confidence: {parsed_doc.confidence_score:.2f}")
            print(f"Entities: {len(parsed_doc.medical_entities)}")
            print(f"Sections: {len(parsed_doc.sections)}")
            
            print("\n🏥 TOP ENTITIES:")
            for entity in sorted(parsed_doc.medical_entities, key=lambda x: x.confidence, reverse=True)[:5]:
                print(f"  {entity.entity_type}: {entity.value} ({entity.confidence:.2f})")
            
            print("\n📝 SUMMARY:")
            print(parsed_doc.summary)
            
            # Save results
            save = input("\nSave results? (y/n): ").lower()
            if save == 'y':
                parser.save_results(parsed_doc, "./interactive_results")
                print("💾 Results saved!")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Main function to run examples."""
    print("🏥 Medical Document Parser Examples")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "medical_note":
            example_medical_note()
        elif command == "lab_report":
            example_lab_report()
        elif command == "radiology":
            example_radiology_report()
        elif command == "batch":
            example_batch_processing()
        elif command == "interactive":
            interactive_parsing()
        else:
            print("Unknown command. Available: medical_note, lab_report, radiology, batch, interactive")
    else:
        # Run all examples
        example_medical_note()
        example_lab_report()
        example_radiology_report()
        example_batch_processing()


if __name__ == "__main__":
    main() 