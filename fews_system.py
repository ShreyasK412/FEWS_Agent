"""
Famine Early Warning System
Core system with 3 functions:
1. Identify at-risk regions from IPC data
2. Explain why (RAG on situation reports)
3. Recommend interventions (RAG on intervention literature)
"""
import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document

from ipc_parser import IPCParser, RegionRiskAssessment
from document_processor import DocumentProcessor


# Setup logging for missing information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('missing_info.log'),
        logging.StreamHandler()
    ]
)
missing_info_logger = logging.getLogger('missing_info')


class FEWSSystem:
    """Famine Early Warning System with 3 core functions."""
    
    def __init__(
        self,
        model_name: str = "llama3.2",
        ipc_file: str = "ipc classification data/ipcFic_data.csv",
        reports_dir: str = "current situation report",
        interventions_dir: str = "intervention-literature"
    ):
        self.model_name = model_name
        self.ipc_file = ipc_file
        self.reports_dir = Path(reports_dir)
        self.interventions_dir = Path(interventions_dir)
        
        # Initialize components
        self.ipc_parser = IPCParser(ipc_file)
        self.doc_processor = DocumentProcessor()
        
        # Vector stores
        self.reports_vectorstore: Optional[Chroma] = None
        self.interventions_vectorstore: Optional[Chroma] = None
        
        # LLM and embeddings
        self.embeddings = None
        self.llm = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and LLM."""
        try:
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            self.llm = OllamaLLM(model=self.model_name)
            print("✅ Initialized LLM and embeddings")
        except Exception as e:
            print(f"⚠️  Error initializing LLM: {e}")
            print("   Make sure Ollama is running: ollama serve")
            print(f"   And model is available: ollama pull {self.model_name}")
    
    def setup_vector_stores(self):
        """Setup separate vector stores for reports and interventions."""
        print("\n" + "="*80)
        print("SETTING UP VECTOR STORES")
        print("="*80)
        
        # Process situation reports
        if self.reports_dir.exists():
            print(f"\n📄 Processing situation reports from: {self.reports_dir}")
            report_files = list(self.reports_dir.glob("*.pdf"))
            if report_files:
                report_chunks = []
                for pdf_file in report_files:
                    chunks = self.doc_processor.process_pdf(str(pdf_file))
                    report_chunks.extend(chunks)
                
                if report_chunks:
                    documents = [
                        Document(
                            page_content=chunk["content"],
                            metadata={"source": chunk["source"], "type": "situation_report"}
                        )
                        for chunk in report_chunks
                    ]
                    
                    self.reports_vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory="./chroma_db_reports"
                    )
                    print(f"✅ Created reports vector store with {len(documents)} chunks")
                else:
                    print("⚠️  No content extracted from reports")
            else:
                print(f"⚠️  No PDF files found in {self.reports_dir}")
        else:
            print(f"⚠️  Reports directory not found: {self.reports_dir}")
        
        # Process intervention literature
        if self.interventions_dir.exists():
            print(f"\n📚 Processing intervention literature from: {self.interventions_dir}")
            intervention_files = list(self.interventions_dir.glob("*.pdf"))
            if intervention_files:
                intervention_chunks = []
                for pdf_file in intervention_files:
                    chunks = self.doc_processor.process_pdf(str(pdf_file))
                    intervention_chunks.extend(chunks)
                
                if intervention_chunks:
                    documents = [
                        Document(
                            page_content=chunk["content"],
                            metadata={"source": chunk["source"], "type": "intervention_literature"}
                        )
                        for chunk in intervention_chunks
                    ]
                    
                    self.interventions_vectorstore = Chroma.from_documents(
                        documents=documents,
                        embedding=self.embeddings,
                        persist_directory="./chroma_db_interventions"
                    )
                    print(f"✅ Created interventions vector store with {len(documents)} chunks")
                else:
                    print("⚠️  No content extracted from intervention literature")
            else:
                print(f"⚠️  No PDF files found in {self.interventions_dir}")
        else:
            print(f"⚠️  Interventions directory not found: {self.interventions_dir}")
    
    def function1_identify_at_risk_regions(self) -> List[RegionRiskAssessment]:
        """
        Function 1: Identify at-risk regions from IPC data.
        
        Returns:
            List of risk assessments, sorted by risk level
        """
        print("\n" + "="*80)
        print("FUNCTION 1: IDENTIFYING AT-RISK REGIONS")
        print("="*80)
        
        assessments = self.ipc_parser.identify_at_risk_regions()
        at_risk = [a for a in assessments if a.is_at_risk]
        
        print(f"\n✅ Identified {len(at_risk)} at-risk regions out of {len(assessments)} total")
        print(f"   High Risk: {len([a for a in at_risk if a.risk_level == 'HIGH'])}")
        print(f"   Medium Risk: {len([a for a in at_risk if a.risk_level == 'MEDIUM'])}")
        print(f"   Low Risk: {len([a for a in at_risk if a.risk_level == 'LOW'])}")
        
        return at_risk
    
    def function2_explain_why(
        self, 
        region: str,
        assessment: Optional[RegionRiskAssessment] = None
    ) -> Dict:
        """
        Function 2: Explain why a region is at risk.
        
        Uses RAG on situation reports to find drivers:
        - Conflict
        - Drought/rainfall
        - Price increases
        - Displacement
        - Other factors
        
        If insufficient data found, logs to missing_info.log.
        
        Args:
            region: Region name
            assessment: Optional risk assessment (will fetch if not provided)
        
        Returns:
            Dict with explanation, drivers, sources, and data_quality
        """
        print("\n" + "="*80)
        print(f"FUNCTION 2: EXPLAINING WHY {region.upper()} IS AT RISK")
        print("="*80)
        
        # Get assessment if not provided
        if assessment is None:
            assessment = self.ipc_parser.get_region_assessment(region)
            if assessment is None:
                return {
                    "region": region,
                    "explanation": f"Region '{region}' not found in IPC data.",
                    "drivers": [],
                    "sources": [],
                    "data_quality": "not_found",
                    "ipc_phase": None
                }
        
        # Build query
        query = f"""
        Food security situation in {region}, Ethiopia. 
        Current IPC phase: {assessment.current_phase}
        What are the main drivers of food insecurity?
        Look for: conflict, drought, rainfall, price increases, displacement, 
        market disruptions, crop failures, livestock losses.
        """
        
        # Retrieve relevant documents
        if self.reports_vectorstore is None:
            explanation = (
                f"Region {region} has IPC Phase {assessment.current_phase}. "
                f"However, I cannot access situation reports to explain the drivers. "
                f"Please ensure situation reports are processed."
            )
            missing_info_logger.warning(
                f"Region: {region} | Issue: No vector store for situation reports"
            )
            return {
                "region": region,
                "explanation": explanation,
                "drivers": [],
                "sources": [],
                "data_quality": "no_vector_store",
                "ipc_phase": assessment.current_phase
            }
        
        try:
            # Retrieve top relevant chunks
            retriever = self.reports_vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)
            
            if not docs or len(docs) == 0:
                explanation = (
                    f"Region {region} has IPC Phase {assessment.current_phase}, "
                    f"indicating {'Emergency' if assessment.current_phase >= 4 else 'Crisis' if assessment.current_phase >= 3 else 'Stressed'} conditions. "
                    f"However, I could not find specific information in the situation reports "
                    f"about what is causing the food insecurity in this region."
                )
                missing_info_logger.warning(
                    f"Region: {region} | IPC Phase: {assessment.current_phase} | "
                    f"Issue: No relevant information found in situation reports"
                )
                return {
                    "region": region,
                    "explanation": explanation,
                    "drivers": [],
                    "sources": [],
                    "data_quality": "insufficient",
                    "ipc_phase": assessment.current_phase
                }
            
            # Build context from retrieved documents
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:500]}"
                for doc in docs
            ])
            
            # Use LLM to extract drivers
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "context"],
                template="""You are a food security analyst. Based on the following context from situation reports, explain why {region} is experiencing IPC Phase {ipc_phase} food insecurity.

CONTEXT FROM SITUATION REPORTS:
{context}

INSTRUCTIONS:
1. Extract the MAIN DRIVERS of food insecurity for {region}
2. Be specific: mention conflict, drought, price increases, displacement, etc.
3. Cite specific information from the context
4. If the context does NOT contain clear information about {region}, explicitly state: "I cannot find specific information about {region} in the available reports."
5. Do NOT make up or infer information that is not in the context.

Provide a clear, structured explanation with:
- Main drivers (conflict, drought, prices, displacement, etc.)
- Specific evidence from the reports
- Any limitations in the available information

EXPLANATION:"""
            )
            
            chain = prompt | self.llm
            explanation = chain.invoke({
                "region": region,
                "ipc_phase": assessment.current_phase,
                "context": context
            })
            
            # Extract drivers (simple keyword extraction)
            drivers = []
            explanation_lower = explanation.lower()
            if "conflict" in explanation_lower or "violence" in explanation_lower:
                drivers.append("Conflict")
            if "drought" in explanation_lower or "rainfall" in explanation_lower or "dry" in explanation_lower:
                drivers.append("Drought/Rainfall deficit")
            if "price" in explanation_lower or "cost" in explanation_lower:
                drivers.append("Price increases")
            if "displacement" in explanation_lower or "displaced" in explanation_lower:
                drivers.append("Displacement")
            if "crop" in explanation_lower or "harvest" in explanation_lower:
                drivers.append("Crop failure")
            if "livestock" in explanation_lower:
                drivers.append("Livestock losses")
            
            # Check if explanation indicates insufficient data
            data_quality = "sufficient"
            if "cannot find" in explanation_lower or "not find" in explanation_lower or "insufficient" in explanation_lower:
                data_quality = "insufficient"
                missing_info_logger.warning(
                    f"Region: {region} | IPC Phase: {assessment.current_phase} | "
                    f"Issue: Insufficient information in situation reports"
                )
            
            sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
            
            return {
                "region": region,
                "explanation": explanation,
                "drivers": drivers if drivers else ["Information not clearly identified"],
                "sources": list(set(sources)),
                "data_quality": data_quality,
                "ipc_phase": assessment.current_phase,
                "retrieved_chunks": len(docs)
            }
            
        except Exception as e:
            error_msg = f"Error retrieving information: {str(e)}"
            missing_info_logger.error(
                f"Region: {region} | Error: {error_msg}"
            )
            return {
                "region": region,
                "explanation": f"Error accessing situation reports: {error_msg}",
                "drivers": [],
                "sources": [],
                "data_quality": "error",
                "ipc_phase": assessment.current_phase
            }
    
    def function3_recommend_interventions(
        self,
        region: str,
        assessment: Optional[RegionRiskAssessment] = None,
        drivers: Optional[List[str]] = None
    ) -> Dict:
        """
        Function 3: Recommend interventions based on IPC phase and drivers.
        
        Uses RAG on intervention literature to generate practical recommendations
        for small NGOs/individuals.
        
        Args:
            region: Region name
            assessment: Optional risk assessment
            drivers: Optional list of drivers (from function 2)
        
        Returns:
            Dict with recommendations, sources, and limitations
        """
        print("\n" + "="*80)
        print(f"FUNCTION 3: RECOMMENDING INTERVENTIONS FOR {region.upper()}")
        print("="*80)
        
        # Get assessment if not provided
        if assessment is None:
            assessment = self.ipc_parser.get_region_assessment(region)
            if assessment is None:
                return {
                    "region": region,
                    "recommendations": f"Region '{region}' not found in IPC data.",
                    "sources": [],
                    "limitations": "Region not found in IPC data"
                }
        
        ipc_phase = assessment.current_phase
        drivers_str = ", ".join(drivers) if drivers else "general food insecurity"
        
        # Build query
        query = f"""
        Interventions for IPC Phase {ipc_phase} food insecurity.
        Region: {region}, Ethiopia
        Drivers: {drivers_str}
        Provide practical recommendations for small NGOs and individual aid workers.
        Focus on: emergency food assistance, cash transfers, nutrition programs, 
        water and sanitation, protection services, livelihood support.
        """
        
        # Retrieve relevant documents
        if self.interventions_vectorstore is None:
            return {
                "region": region,
                "recommendations": (
                    f"For {region} with IPC Phase {ipc_phase}, I cannot access "
                    f"intervention literature. Please ensure intervention documents are processed."
                ),
                "sources": [],
                "limitations": "No intervention vector store available"
            }
        
        try:
            # Retrieve top relevant chunks
            retriever = self.interventions_vectorstore.as_retriever(search_kwargs={"k": 5})
            docs = retriever.get_relevant_documents(query)
            
            if not docs or len(docs) == 0:
                return {
                    "region": region,
                    "recommendations": (
                        f"For {region} with IPC Phase {ipc_phase}, I could not find "
                        f"specific intervention guidance in the available literature."
                    ),
                    "sources": [],
                    "limitations": "No relevant intervention guidance found"
                }
            
            # Build context
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:500]}"
                for doc in docs
            ])
            
            # Use LLM to generate recommendations
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "drivers", "context"],
                template="""You are a humanitarian response advisor. Based on intervention literature, provide practical recommendations for addressing food insecurity in {region}, Ethiopia.

REGION: {region}
IPC PHASE: {ipc_phase} ({'Emergency' if ipc_phase >= 4 else 'Crisis' if ipc_phase >= 3 else 'Stressed'})
DRIVERS: {drivers}

INTERVENTION LITERATURE:
{context}

INSTRUCTIONS:
1. Provide SPECIFIC, PRACTICAL recommendations suitable for small NGOs and individual aid workers
2. Base recommendations on the intervention literature provided
3. Consider the IPC phase and drivers
4. Include: immediate actions, medium-term interventions, coordination needs
5. Cite specific sources from the literature
6. If the literature does NOT contain sufficient guidance, explicitly state the limitations
7. Do NOT make up interventions not supported by the literature

Provide structured recommendations with:
- Immediate humanitarian response
- Medium-term resilience building
- Coordination and logistics considerations
- Any limitations in available guidance

RECOMMENDATIONS:"""
            )
            
            chain = prompt | self.llm
            recommendations = chain.invoke({
                "region": region,
                "ipc_phase": ipc_phase,
                "drivers": drivers_str,
                "context": context
            })
            
            sources = [doc.metadata.get('source', 'Unknown') for doc in docs]
            
            # Check for limitations
            limitations = None
            if "limitation" in recommendations.lower() or "cannot" in recommendations.lower():
                limitations = "Some limitations noted in recommendations"
            
            return {
                "region": region,
                "recommendations": recommendations,
                "sources": list(set(sources)),
                "limitations": limitations,
                "ipc_phase": ipc_phase,
                "retrieved_chunks": len(docs)
            }
            
        except Exception as e:
            return {
                "region": region,
                "recommendations": f"Error accessing intervention literature: {str(e)}",
                "sources": [],
                "limitations": f"Error: {str(e)}"
            }


if __name__ == "__main__":
    # Test the system
    system = FEWSSystem()
    
    # Setup vector stores
    system.setup_vector_stores()
    
    # Function 1: Identify at-risk regions
    at_risk = system.function1_identify_at_risk_regions()
    
    if at_risk:
        # Test Function 2: Explain why
        test_region = at_risk[0].region
        explanation = system.function2_explain_why(test_region, at_risk[0])
        print(f"\nExplanation for {test_region}:")
        print(explanation["explanation"])
        
        # Test Function 3: Recommend interventions
        interventions = system.function3_recommend_interventions(
            test_region, 
            at_risk[0], 
            explanation.get("drivers", [])
        )
        print(f"\nInterventions for {test_region}:")
        print(interventions["recommendations"])

