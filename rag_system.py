"""
RAG system with vector store and retrieval.
Uses ChromaDB for vector storage and Ollama for LLM.
"""

import os
from typing import List, Dict
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from intent_detector import IntentDetector


class RAGSystem:
    """RAG system for querying documents and price data."""
    
    def __init__(self, model_name: str = "llama3.2", persist_directory: str = "./chroma_db"):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.intent_detector = IntentDetector()
        
    def initialize(self):
        """Initialize embeddings and LLM."""
        print("Initializing embeddings and LLM...")
        try:
            # Initialize Ollama embeddings
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            print(f"✓ Embeddings initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            print("Make sure Ollama is running and the model is available.")
            raise
        
        try:
            # Initialize Ollama LLM
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✓ LLM initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            print("Make sure Ollama is running and the model is available.")
            raise
    
    def create_vector_store(self, chunks: List[Dict[str, str]]):
        """Create vector store from document chunks."""
        print(f"\nCreating vector store from {len(chunks)} chunks...")
        
        # Convert chunks to LangChain documents
        print("Converting chunks to documents...")
        documents = []
        for i, chunk in enumerate(chunks):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "type": chunk["type"]
                }
            )
            documents.append(doc)
        
        # Create vector store
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading existing vector store...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                existing_count = self.vectorstore._collection.count()
                print(f"  Found {existing_count} existing documents in vector store")
                
                # Check if we need to add documents
                if existing_count == 0:
                    print("  Vector store is empty, adding all documents (this will take a while)...")
                    print("  Generating embeddings for documents (this may take 10-20 minutes)...")
                    self.vectorstore.add_documents(documents)
                    print("✓ Added documents to vector store")
                else:
                    print(f"  Vector store already has {existing_count} documents.")
                    print("  Skipping document addition (assuming documents already processed).")
                    print("  If you added new files, delete chroma_db/ and restart to reprocess.")
                    
            except Exception as e:
                print(f"  Error loading existing store: {e}")
                print("  Creating new vector store...")
                print("  Generating embeddings for documents (this may take 10-20 minutes)...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
                print("✓ Vector store created")
        else:
            print("Creating new vector store...")
            print("  Generating embeddings for documents (this may take 10-20 minutes)...")
            print("  This is normal for the first run - embeddings are cached for future runs.")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
            print("✓ Vector store created")
        
        final_count = self.vectorstore._collection.count()
        print(f"\n✓ Vector store ready with {final_count} documents")
    
    def setup_qa_chain(self, k: int = 4):
        """Setup the QA chain with custom prompt."""
        
        # Custom prompt with humanitarian-response decision playbook
        prompt_template = """You are an AI assistant for Ethiopian food-security decision support.
Your audience includes analysts, policymakers, and humanitarian actors.
Maintain a neutral, factual, policy-analytic tone.

You have access to research documents and food price data from the World Food Programme.

CONTEXT:
{context}

QUESTION: {question}

RESPONSE INSTRUCTIONS:

1. **Determine Query Type:**
   - If query asks about prices/markets/numeric data: Provide factual price data with 1-line interpretation
   - If query asks about situation/context/"what should we do"/"how can we help": Use structured response framework below
   - If query is hypothetical/scenario: Mark clearly as scenario-based and use historical parallels

2. **For Price/Market Queries:**
   Format: "Commodity in Region (Date): Price ETB/unit (+% change if applicable) — brief interpretation."
   Example: "Maize in Oromia (Jul 2023): 8,000 ETB/100 kg (+18%)—reflects transport disruption."

3. **For Situation/Context/Humanitarian Response Queries:**
   Structure your response as follows:

   **Situation Overview:** 2-3 data-grounded sentences summarizing crisis scope and quantitative indicators (funding gaps, rations, IPC phases, market inflation, displacement numbers).

   **Operational Implications:** 2-4 sentences explaining drivers and consequences (supply disruptions, malnutrition trends, displacement, market dynamics, access constraints).

   **Recommended Actions:** Categorize under:
   - **Immediate Humanitarian:** food distribution, nutrition programs, logistics, fuel corridors, ration restoration
   - **Medium-Term Resilience:** cash transfers, market support, agricultural inputs, livelihoods, local capacity building
   - **Policy / Coordination:** funding advocacy, donor engagement, early-warning monitoring, coordination mechanisms

   For analysts/policymakers: End with a decision takeaway (e.g., "These trends imply urgent donor mobilization and contingency planning for IPC 4 hotspots.")

   For public/NGO users: Add vetted support channels (WFP, UNHCR, Save the Children, ACF) and verified local partners.

4. **For Scenario/Hypothetical Queries:**
   - Clearly mark as scenario-based: "Based on scenario analysis..."
   - Use historical parallels from context
   - Provide conditional projections with clear assumptions

5. **General Rules:**
   - Always ground responses in retrieved data
   - Cite specific data points (prices, dates, locations, commodities, funding amounts)
   - If no data found: "No current data found in famine-watch sources."
   - If extrapolating: "Based on past conditions..."
   - Ask for missing region/commodity if needed for specificity

6. **Formatting:**
   - Use clear section headers (## or **bold**)
   - Use bullet points for actions
   - Include quantitative indicators where available
   - Keep concise but comprehensive

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create retriever with default k (will be adjusted per query based on intent)
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": k}
        )
        
        # Store retriever for dynamic updates
        self.default_retriever = retriever
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        
        print("✓ QA chain setup complete")
    
    def query(self, question: str, intent_info: Dict = None) -> Dict:
        """
        Query the RAG system.
        
        Args:
            question: The query string
            intent_info: Optional intent detection results (if not provided, will be detected)
        
        Returns:
            Dict with result, sources, and intent information
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        # Detect intent if not provided
        if intent_info is None:
            intent_info = self.intent_detector.detect_intent(question)
        
        # Adjust retrieval based on intent - update retriever dynamically
        k = 6 if intent_info['needs_structured_response'] else 4
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        # Temporarily update the QA chain's retriever
        original_retriever = self.qa_chain.retriever
        self.qa_chain.retriever = retriever
        
        try:
            # Query with enhanced context
            result = self.qa_chain.invoke({"query": question})
        finally:
            # Restore original retriever
            self.qa_chain.retriever = original_retriever
        
        # Add intent info to result
        result['intent_info'] = intent_info
        
        return result
    
    def get_relevant_chunks(self, question: str, k: int = 4) -> List[Document]:
        """Get relevant chunks for a question without generating an answer."""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized.")
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs

