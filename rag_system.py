"""
Unified RAG system for food security analysis.
Combines document retrieval with risk prediction and intervention recommendations.
"""
import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from intent_detector import IntentDetector
from models.risk_predictor import EnsemblePredictor, RiskAssessment
from models.feature_engineering import FeatureEngineer
from models.intervention_recommender import InterventionRecommender
from data_loader import DataLoader


class RAGSystem:
    """Unified RAG system with risk identification and intervention recommendations."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        persist_directory: str = "./chroma_db",
        documents_dir: str = "documents"
    ):
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.documents_dir = documents_dir
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.intent_detector = IntentDetector()
        self.data_loader = DataLoader()
        self.risk_predictor = EnsemblePredictor()
        self.feature_engineer = FeatureEngineer()
        self.intervention_recommender = InterventionRecommender()
    
    def initialize(self):
        """Initialize embeddings and LLM."""
        print("Initializing RAG system...")
        try:
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            print(f"✓ Embeddings initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing embeddings: {e}")
            raise
        
        try:
            self.llm = OllamaLLM(model=self.model_name)
            print(f"✓ LLM initialized with model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def create_vector_store(self, chunks: List[Dict[str, str]]):
        """Create vector store from document chunks."""
        print(f"\nCreating vector store from {len(chunks)} chunks...")
        
        documents = []
        for i, chunk in enumerate(chunks):
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(chunks)} chunks...")
            doc = Document(
                page_content=chunk["content"],
                metadata={
                    "source": chunk["source"],
                    "chunk_id": chunk.get("chunk_id", i),
                    "type": chunk.get("type", "unknown")
                }
            )
            documents.append(doc)
        
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            print("Loading existing vector store...")
            try:
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embeddings
                )
                existing_count = self.vectorstore._collection.count()
                if existing_count == 0:
                    print("  Adding documents (this may take a while)...")
                    self.vectorstore.add_documents(documents)
                else:
                    print(f"  Vector store already has {existing_count} documents.")
            except Exception as e:
                print(f"  Error loading store: {e}")
                print("  Creating new vector store...")
                self.vectorstore = Chroma.from_documents(
                    documents=documents,
                    embedding=self.embeddings,
                    persist_directory=self.persist_directory
                )
        else:
            print("Creating new vector store...")
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        
        final_count = self.vectorstore._collection.count()
        print(f"\n✓ Vector store ready with {final_count} documents")
    
    def load_vector_store(self):
        """Load existing vector store."""
        if os.path.exists(self.persist_directory) and os.listdir(self.persist_directory):
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings
            )
            count = self.vectorstore._collection.count()
            print(f"✓ Loaded vector store with {count} documents")
        else:
            raise ValueError("Vector store not found. Process documents first.")
    
    def setup_qa_chain(self, k: int = 6):
        """Setup QA chain with food security prompts."""
        prompt_template = """You are an expert food security analyst specializing in early warning and intervention planning.
Your audience includes analysts, policymakers, and humanitarian actors.

CONTEXT:
{context}

QUESTION: {question}

RESPONSE INSTRUCTIONS:

1. **For Risk Assessment Queries:**
   Provide structured risk analysis:
   - Risk Level: HIGH/MEDIUM/LOW
   - Predicted IPC Phase: 1-5
   - Key Drivers: List primary risk factors
   - Confidence: Assessment confidence level
   - Time Horizon: Prediction period
   - Population at Risk: If available

2. **For Intervention Queries:**
   Provide prioritized intervention recommendations:
   - Immediate Actions: Urgent humanitarian response
   - Medium-Term Actions: Resilience and recovery
   - Policy Actions: Coordination and advocacy
   - Evidence: Cite sources from context
   - Feasibility: Consider regional constraints

3. **For Price/Market Queries:**
   Format: "Commodity in Region (Date): Price ETB/unit (+% change if applicable) — brief interpretation."

4. **General Rules:**
   - Always ground responses in retrieved data
   - Cite specific sources and dates
   - Use IPC 5-phase scale correctly
   - Provide actionable recommendations
   - Mark uncertainty when data is limited

Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def query(self, question: str, intent_info: Dict = None) -> Dict:
        """
        General query method (for backward compatibility).
        """
        if not self.qa_chain:
            raise ValueError("QA chain not initialized. Call setup_qa_chain() first.")
        
        if intent_info is None:
            intent_info = self.intent_detector.detect_intent(question)
        
        k = 6 if intent_info['needs_structured_response'] else 4
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        
        original_retriever = self.qa_chain.retriever
        self.qa_chain.retriever = retriever
        
        try:
            result = self.qa_chain.invoke({"query": question})
        finally:
            self.qa_chain.retriever = original_retriever
        
        result['intent_info'] = intent_info
        return result
    
    def identify_at_risk_regions(
        self,
        country: str = "ETH",
        time_horizon: str = "3 months"
    ) -> List[RiskAssessment]:
        """
        Function 1: Identify regions at risk of food insecurity.
        
        Steps:
        1. Load data from CSV files
        2. Engineer features
        3. Run risk prediction models
        4. Retrieve document context for synthesis
        5. Return ranked list of at-risk regions
        """
        print("\n" + "="*80)
        print("IDENTIFYING AT-RISK REGIONS")
        print("="*80)
        print()
        
        # Step 1: Load data
        unified_data = self.data_loader.create_unified_dataset()
        
        if unified_data is None or len(unified_data) == 0:
            print("❌ No data available for risk assessment")
            print("   Upload IPC phase data to: data/raw/ipc/ethiopia/ipc_phases.csv")
            return []
        
        # Step 2: Engineer features
        print("\nEngineering features...")
        features = self.feature_engineer.engineer_all_features(unified_data)
        
        # Step 3: Predict risk for each region
        print("\nRunning risk predictions...")
        regions = features["admin2"].unique() if "admin2" in features.columns else features["region"].unique()
        assessments = []
        
        for region in regions:
            assessment = self.risk_predictor.predict(features, region)
            
            # Step 4: Enhance with document context
            if self.vectorstore:
                query = f"Food security situation in {region}, Ethiopia. Current IPC phase, key drivers, population affected."
                doc_context = self._retrieve_context(query, top_k=3)
                
                # Use LLM to synthesize if context available
                if doc_context and self.qa_chain:
                    synthesis_query = f"""
Based on the following data and documents, provide a risk assessment for {region}:

Predicted Risk: {assessment.risk_level}
IPC Phase: {assessment.ipc_phase_prediction}
Key Drivers: {', '.join(assessment.key_drivers)}

Document Context:
{chr(10).join([d['text'][:300] for d in doc_context])}

Provide a brief risk assessment summary.
"""
                    try:
                        result = self.qa_chain.invoke({"query": synthesis_query})
                        # Could parse result to enhance assessment
                    except:
                        pass
            
            assessments.append(assessment)
        
        # Sort by risk level (HIGH first)
        risk_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1, "UNKNOWN": 0}
        assessments.sort(key=lambda x: (risk_order.get(x.risk_level, 0), x.confidence_score), reverse=True)
        
        print(f"\n✓ Identified {len(assessments)} regions")
        print(f"  High risk: {sum(1 for a in assessments if a.risk_level == 'HIGH')}")
        print(f"  Medium risk: {sum(1 for a in assessments if a.risk_level == 'MEDIUM')}")
        print(f"  Low risk: {sum(1 for a in assessments if a.risk_level == 'LOW')}")
        
        return assessments
    
    def recommend_interventions(
        self,
        region: str,
        risk_assessment: Optional[RiskAssessment] = None
    ) -> Dict:
        """
        Function 2: Recommend intervention steps for a region.
        
        Steps:
        1. Get risk assessment (or generate if not provided)
        2. Retrieve similar historical situations from documents
        3. Map risk drivers to interventions
        4. Generate prioritized intervention plan
        """
        print("\n" + "="*80)
        print(f"RECOMMENDING INTERVENTIONS FOR {region.upper()}")
        print("="*80)
        print()
        
        # Step 1: Get risk assessment if not provided
        if risk_assessment is None:
            print("Generating risk assessment first...")
            assessments = self.identify_at_risk_regions()
            risk_assessment = next((a for a in assessments if a.region_id == region), None)
            
            if risk_assessment is None:
                # Create default assessment
                risk_assessment = RiskAssessment(
                    region_id=region,
                    risk_level="MEDIUM",
                    ipc_phase_prediction=3,
                    confidence_score=0.5,
                    key_drivers=["Insufficient data"],
                    time_horizon="3 months"
                )
        
        # Step 2: Retrieve document context
        context_docs = []
        if self.vectorstore:
            query = f"""
Intervention strategies for {region} at IPC Phase {risk_assessment.ipc_phase_prediction}.
Key drivers: {', '.join(risk_assessment.key_drivers)}.
What interventions were successful in similar situations?
"""
            context_docs = self._retrieve_context(query, top_k=5)
        
        # Step 3: Get intervention recommendations
        recommendations = self.intervention_recommender.recommend(
            risk_assessment,
            document_context=context_docs
        )
        
        # Step 4: Generate detailed intervention plan using LLM
        intervention_text = ""
        if self.qa_chain and context_docs:
            intervention_prompt = f"""
Region: {risk_assessment.region_id}
Risk Level: {risk_assessment.risk_level}
IPC Phase: {risk_assessment.ipc_phase_prediction}
Key Drivers: {', '.join(risk_assessment.key_drivers)}

Historical Context from Documents:
{chr(10).join([d['text'][:400] for d in context_docs])}

Based on this information, provide a detailed intervention plan with:
1. Immediate humanitarian actions (next 1-3 months)
2. Medium-term resilience building (3-12 months)
3. Policy and coordination needs
4. Resource requirements
5. Success indicators and monitoring

Be specific and actionable.
"""
            try:
                result = self.qa_chain.invoke({"query": intervention_prompt})
                intervention_text = result["result"]
            except Exception as e:
                print(f"⚠️  Error generating intervention plan: {e}")
                intervention_text = "Intervention plan generation failed. Use recommendations above."
        else:
            intervention_text = "Intervention recommendations based on risk drivers:\n"
            for rec in recommendations:
                intervention_text += f"- {rec.get('name', 'Unknown intervention')}\n"
        
        return {
            "region": risk_assessment.region_id,
            "risk_assessment": risk_assessment,
            "interventions": intervention_text,
            "recommendations": recommendations,
            "sources": [d.get("source", "Unknown") for d in context_docs] if context_docs else [],
            "confidence": risk_assessment.confidence_score
        }
    
    def _retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve relevant context from vector store."""
        if not self.vectorstore:
            return []
        
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        docs = retriever.get_relevant_documents(query)
        
        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "Unknown"),
                "type": doc.metadata.get("type", "Unknown")
            }
            for doc in docs
        ]


if __name__ == "__main__":
    # Test
    rag = RAGSystem()
    rag.initialize()
    rag.load_vector_store()
    rag.setup_qa_chain()
    
    # Test risk identification
    assessments = rag.identify_at_risk_regions()
    print(f"\nFound {len(assessments)} regions")
    
    # Test intervention recommendation
    if assessments:
        intervention = rag.recommend_interventions(assessments[0].region_id, assessments[0])
        print(f"\nIntervention plan for {intervention['region']}")
