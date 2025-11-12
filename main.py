"""
Unified main entry point for FEWS Agent.
Two core functions:
1. Identify regions at risk
2. Recommend intervention steps
"""
import os
import sys
from pathlib import Path
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from models.risk_predictor import RiskAssessment


def check_ollama():
    """Check if Ollama is available."""
    try:
        import ollama
        models = ollama.list()
        if not models.get('models', []):
            print("Warning: Ollama is running but no models found.")
            print("Please pull a model first: ollama pull llama3.2")
            return False
        return True
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed: https://ollama.ai/download")
        print("2. Ollama is running: ollama serve")
        print("3. A model is pulled: ollama pull llama3.2")
        return False


def main():
    """Main function with unified system."""
    print("="*80)
    print("FEWS AGENT: FOOD SECURITY EARLY WARNING SYSTEM")
    print("="*80)
    print("\nCore Functions:")
    print("  1. Identify regions at risk")
    print("  2. Recommend intervention steps")
    print("\n" + "="*80)
    print("\nInitializing system...\n")
    
    # Check Ollama
    if not check_ollama():
        sys.exit(1)
    
    # Configuration
    MODEL_NAME = os.getenv("FEWS_MODEL_NAME", "llama3.2")
    DOCUMENTS_DIR = "documents"
    CHROMA_DB_DIR = "./chroma_db"
    
    # Step 1: Process documents if needed
    print("Step 1: Checking documents...")
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        print("  Processing documents...")
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
        
        if not pdf_files:
            print(f"  ⚠️  No PDFs found in {DOCUMENTS_DIR}/")
            print("  Documents are optional but recommended for better context")
        else:
            chunks = []
            for pdf_file in pdf_files:
                chunks.extend(processor.process_pdf(str(pdf_file)))
            
            if chunks:
                print(f"  Processed {len(chunks)} chunks from {len(pdf_files)} PDFs")
            else:
                print("  ⚠️  No chunks extracted from PDFs")
    else:
        print("  ✓ Documents already processed")
    
    # Step 2: Initialize unified RAG system
    print(f"\nStep 2: Initializing RAG system with model '{MODEL_NAME}'...")
    rag = RAGSystem(
        model_name=MODEL_NAME,
        persist_directory=CHROMA_DB_DIR,
        documents_dir=DOCUMENTS_DIR
    )
    
    try:
        rag.initialize()
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        sys.exit(1)
    
    # Step 3: Load vector store
    print(f"\nStep 3: Loading vector store...")
    try:
        rag.load_vector_store()
    except:
        # Create new vector store if needed
        if os.path.exists(DOCUMENTS_DIR):
            processor = DocumentProcessor()
            pdf_files = list(Path(DOCUMENTS_DIR).glob("*.pdf"))
            chunks = []
            for pdf_file in pdf_files:
                chunks.extend(processor.process_pdf(str(pdf_file)))
            
            if chunks:
                rag.create_vector_store(chunks)
                print(f"  ✓ Created vector store with {len(chunks)} chunks")
            else:
                print("  ⚠️  No documents to process")
        else:
            print("  ⚠️  Documents directory not found")
    
    # Step 4: Setup QA chain
    print(f"\nStep 4: Setting up QA chain...")
    rag.setup_qa_chain(k=6)
    
    print("\n" + "="*80)
    print("✓ SYSTEM READY")
    print("="*80)
    print("\nAvailable Commands:")
    print("  'risk' or 'identify risk' - Identify at-risk regions")
    print("  'interventions [region]' - Get intervention steps for a region")
    print("  'help' - Show example queries")
    print("  'exit' - Quit")
    print("\n" + "-"*80 + "\n")
    
    # Interactive loop
    while True:
        try:
            query = input("Query: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("\nGoodbye!")
                break
            
            if query.lower() == 'help':
                print("\nExample queries:")
                print("  - 'risk' - Identify all at-risk regions")
                print("  - 'interventions Tigray' - Get intervention steps for Tigray")
                print("  - 'interventions Amhara' - Get intervention steps for Amhara")
                print("  - 'What are current maize prices?' - General query")
                print()
                continue
            
            # Function 1: Identify at-risk regions
            if query.lower() in ['risk', 'identify risk', 'at risk', 'regions at risk']:
                assessments = rag.identify_at_risk_regions()
                
                if assessments:
                    print("\n" + "="*80)
                    print("AT-RISK REGIONS")
                    print("="*80)
                    print()
                    
                    high_risk = [a for a in assessments if a.risk_level == "HIGH"]
                    medium_risk = [a for a in assessments if a.risk_level == "MEDIUM"]
                    low_risk = [a for a in assessments if a.risk_level == "LOW"]
                    
                    if high_risk:
                        print("🔴 HIGH RISK REGIONS:")
                        for i, assessment in enumerate(high_risk, 1):
                            print(f"\n{i}. {assessment.region_id}")
                            print(f"   IPC Phase: {assessment.ipc_phase_prediction}")
                            print(f"   Confidence: {assessment.confidence_score:.0%}")
                            print(f"   Key Drivers: {', '.join(assessment.key_drivers)}")
                            if assessment.population_at_risk:
                                print(f"   Population at Risk: {assessment.population_at_risk:,}")
                    
                    if medium_risk:
                        print("\n🟡 MEDIUM RISK REGIONS:")
                        for i, assessment in enumerate(medium_risk[:5], 1):  # Top 5
                            print(f"\n{i}. {assessment.region_id} (IPC Phase {assessment.ipc_phase_prediction})")
                    
                    if low_risk:
                        print(f"\n🟢 LOW RISK: {len(low_risk)} regions")
                    
                    print("\n" + "="*80 + "\n")
                else:
                    print("\n⚠️  No risk assessments generated. Check data files.\n")
            
            # Function 2: Recommend interventions
            elif query.lower().startswith('interventions'):
                parts = query.split()
                if len(parts) > 1:
                    region = ' '.join(parts[1:])
                else:
                    region = input("Enter region name: ").strip()
                
                if not region:
                    print("⚠️  Please specify a region\n")
                    continue
                
                result = rag.recommend_interventions(region)
                
                print("\n" + "="*80)
                print(f"INTERVENTION RECOMMENDATIONS: {result['region'].upper()}")
                print("="*80)
                print(f"\nRisk Level: {result['risk_assessment'].risk_level}")
                print(f"IPC Phase: {result['risk_assessment'].ipc_phase_prediction}")
                print(f"Key Drivers: {', '.join(result['risk_assessment'].key_drivers)}")
                print("\n" + "-"*80)
                print("\nINTERVENTION PLAN:")
                print("-"*80)
                print(result['interventions'])
                
                if result['sources']:
                    print("\n" + "-"*80)
                    print("SOURCES:")
                    print("-"*80)
                    for source in result['sources'][:3]:
                        print(f"  - {source}")
                
                print("\n" + "="*80 + "\n")
            
            else:
                # General query (use RAG)
                if rag.qa_chain:
                    print("\nProcessing query...")
                    result = rag.query(query)
                    print("\n" + "="*80)
                    print("ANSWER:")
                    print("="*80)
                    print(result["result"])
                    print("\n" + "-"*80)
                    print("SOURCES:")
                    print("-"*80)
                    sources = result.get("source_documents", [])
                    for i, doc in enumerate(sources, 1):
                        source = doc.metadata.get("source", "Unknown")
                        print(f"[{i}] {source}")
                    print("="*80 + "\n")
                else:
                    print("⚠️  RAG system not fully initialized\n")
        
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}\n")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
