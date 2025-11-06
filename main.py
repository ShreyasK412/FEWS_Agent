"""
Main entry point for the Food Insecurity Analysis System.
"""

import os
import sys
from pathlib import Path
from document_processor import DocumentProcessor
from rag_system import RAGSystem
from query_interface import QueryInterface


def check_ollama():
    """Check if Ollama is available."""
    try:
        import ollama
        # Try to list models to check if Ollama is running
        try:
            models_response = ollama.list()
            models = models_response.get('models', []) if isinstance(models_response, dict) else models_response
            if not models:
                print("Warning: Ollama is running but no models found.")
                print("Please pull a model first: ollama pull llama3.2")
                return False
            return True
        except Exception as e:
            print(f"Error connecting to Ollama service: {e}")
            print("\nPlease ensure:")
            print("1. Ollama is installed: https://ollama.ai/download")
            print("2. Ollama is running: ollama serve")
            print("3. A model is pulled: ollama pull llama3.2")
            return False
    except ImportError:
        print("Error: ollama package not installed.")
        print("Please install: pip install ollama")
        return False
    except Exception as e:
        print(f"Error connecting to Ollama: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is installed: https://ollama.ai/download")
        print("2. Ollama is running: ollama serve")
        print("3. A model is pulled: ollama pull llama3.2")
        return False


def main():
    """Main function."""
    print("="*80)
    print("FOOD INSECURITY ANALYSIS SYSTEM FOR ETHIOPIA")
    print("="*80)
    print("\nInitializing system...\n")
    
    # Check Ollama
    if not check_ollama():
        sys.exit(1)
    
    # Configuration
    MODEL_NAME = "llama3.2"  # Change to your preferred model
    DATA_DIRECTORY = "."  # Current directory
    CHROMA_DB_DIR = "./chroma_db"
    
    # Initialize components
    print("Step 1: Processing documents...")
    processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
    chunks = processor.process_all_documents(DATA_DIRECTORY)
    
    if not chunks:
        print("\nError: No documents processed. Please ensure PDFs and CSV files are in the directory.")
        sys.exit(1)
    
    print(f"\nStep 2: Initializing RAG system with model '{MODEL_NAME}'...")
    rag = RAGSystem(model_name=MODEL_NAME, persist_directory=CHROMA_DB_DIR)
    
    try:
        rag.initialize()
    except Exception as e:
        print(f"\nError initializing RAG system: {e}")
        sys.exit(1)
    
    print(f"\nStep 3: Creating vector store...")
    rag.create_vector_store(chunks)
    
    print(f"\nStep 4: Setting up QA chain...")
    rag.setup_qa_chain(k=4)
    
    print("\n✓ System ready!")
    
    # Start interactive interface
    interface = QueryInterface(rag)
    interface.run_interactive()


if __name__ == "__main__":
    main()

