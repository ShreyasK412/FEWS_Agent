"""
Flask web application for the Food Insecurity Analysis System.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import sys
import os

# Import our RAG system
from rag_system import RAGSystem
from document_processor import DocumentProcessor
from models.risk_predictor import RiskAssessment

app = Flask(__name__)
CORS(app)

# Global RAG system instance
rag_system = None

def initialize_rag_system():
    """Initialize the RAG system (called once at startup)."""
    global rag_system
    
    print("Initializing RAG system for web app...")
    
    # Configuration
    MODEL_NAME = "llama3.2"
    DATA_DIRECTORY = "documents"  # PDFs are now in documents/ folder
    CHROMA_DB_DIR = "./chroma_db"
    
    # Check if vector store exists
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        print("Vector store not found. Processing documents first...")
        processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
        chunks = processor.process_all_documents(DATA_DIRECTORY)
        
        if not chunks:
            print("Error: No documents found!")
            return False
        
        rag = RAGSystem(model_name=MODEL_NAME, persist_directory=CHROMA_DB_DIR)
        rag.initialize()
        rag.create_vector_store(chunks)
        rag.setup_qa_chain(k=4)
        rag_system = rag
    else:
        print("Loading existing vector store...")
        rag = RAGSystem(model_name=MODEL_NAME, persist_directory=CHROMA_DB_DIR)
        rag.initialize()
        
        # Check if vector store needs to be created
        try:
            from langchain_chroma import Chroma
            test_store = Chroma(
                persist_directory=CHROMA_DB_DIR,
                embedding_function=rag.embeddings
            )
            count = test_store._collection.count()
            if count == 0:
                print("Vector store is empty. Processing documents...")
                processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
                chunks = processor.process_all_documents(DATA_DIRECTORY)
                if chunks:
                    rag.create_vector_store(chunks)
                else:
                    print("No documents to process!")
                    return False
            else:
                print(f"Found {count} documents in vector store")
                # Use existing vector store
                rag.vectorstore = test_store
        except Exception as e:
            print(f"Error checking vector store: {e}")
            print("Attempting to create new vector store...")
            processor = DocumentProcessor(chunk_size=1000, chunk_overlap=200)
            chunks = processor.process_all_documents(DATA_DIRECTORY)
            if chunks:
                rag.create_vector_store(chunks)
            else:
                return False
        
        rag.setup_qa_chain(k=4)
        rag_system = rag
    
    print("✓ RAG system ready!")
    return True

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def query():
    """Handle query requests."""
    global rag_system
    
    if not rag_system:
        return jsonify({
            'error': 'RAG system not initialized',
            'answer': 'Please wait for the system to initialize...'
        }), 503
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'error': 'No question provided'
            }), 400
        
        # Check for special commands
        question_lower = question.lower()
        
        # Function 1: Identify at-risk regions
        if question_lower in ['risk', 'identify risk', 'at risk', 'regions at risk']:
            assessments = rag_system.identify_at_risk_regions()
            
            # Format assessments for JSON
            assessments_data = []
            for a in assessments:
                assessments_data.append({
                    'region': a.region_id,
                    'risk_level': a.risk_level,
                    'ipc_phase': a.ipc_phase_prediction,
                    'confidence': a.confidence_score,
                    'key_drivers': a.key_drivers,
                    'population_at_risk': a.population_at_risk
                })
            
            return jsonify({
                'answer': f'Identified {len(assessments)} regions',
                'assessments': assessments_data,
                'type': 'risk_assessment'
            })
        
        # Function 2: Recommend interventions
        elif question_lower.startswith('interventions'):
            parts = question_lower.split()
            region = ' '.join(parts[1:]) if len(parts) > 1 else None
            
            if not region:
                return jsonify({
                    'error': 'Please specify a region',
                    'answer': 'Usage: "interventions Tigray" or "interventions Amhara"'
                }), 400
            
            result = rag_system.recommend_interventions(region)
            
            return jsonify({
                'answer': result['interventions'],
                'region': result['region'],
                'risk_assessment': {
                    'risk_level': result['risk_assessment'].risk_level,
                    'ipc_phase': result['risk_assessment'].ipc_phase_prediction,
                    'key_drivers': result['risk_assessment'].key_drivers
                },
                'sources': result['sources'],
                'type': 'interventions'
            })
        
        # General query
        else:
            result = rag_system.query(question)
            
            # Get intent info
            intent_info = result.get('intent_info', {})
            intent = intent_info.get('intent', 'unknown')
            is_scenario = intent_info.get('is_scenario', False)
            
            # Format response
            response = {
                'answer': result.get('result', ''),
                'sources': [],
                'intent': intent,
                'is_scenario': is_scenario,
                'type': 'general'
            }
            
            # Extract sources
            sources = result.get('source_documents', [])
            for doc in sources:
                response['sources'].append({
                    'source': doc.metadata.get('source', 'Unknown'),
                    'type': doc.metadata.get('type', 'Unknown'),
                    'preview': doc.page_content[:200] + '...' if len(doc.page_content) > 200 else doc.page_content
                })
            
            return jsonify(response)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'answer': f'Error processing query: {str(e)}'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint."""
    global rag_system
    return jsonify({
        'status': 'ready' if rag_system else 'initializing',
        'rag_initialized': rag_system is not None
    })

if __name__ == '__main__':
    # Initialize RAG system
    if not initialize_rag_system():
        print("Failed to initialize RAG system. Exiting...")
        sys.exit(1)
    
    # Run the Flask app
    print("\n" + "="*80)
    print("Starting web server...")
    print("Open your browser to: http://localhost:5001")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)

