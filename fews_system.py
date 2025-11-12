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
from pypdf import PdfReader

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
        self.ipc_file = Path(ipc_file)
        self.reports_dir = Path(reports_dir)
        self.interventions_dir = Path(interventions_dir)
        
        # Validate data sources
        print("="*80)
        print("FEWS SYSTEM - DATA SOURCE VALIDATION")
        print("="*80)
        print(f"\n✅ IPC Data: {self.ipc_file}")
        if not self.ipc_file.exists():
            print(f"   ⚠️  WARNING: IPC file not found at {self.ipc_file}")
        else:
            print(f"   ✅ Found")
        
        print(f"\n✅ Situation Reports: {self.reports_dir}")
        if not self.reports_dir.exists():
            print(f"   ⚠️  WARNING: Reports directory not found at {self.reports_dir}")
        else:
            pdf_count = len(list(self.reports_dir.glob("*.pdf")))
            print(f"   ✅ Found {pdf_count} PDF file(s)")
        
        print(f"\n✅ Intervention Literature: {self.interventions_dir}")
        if not self.interventions_dir.exists():
            print(f"   ⚠️  WARNING: Interventions directory not found at {self.interventions_dir}")
        else:
            pdf_count = len(list(self.interventions_dir.glob("*.pdf")))
            print(f"   ✅ Found {pdf_count} PDF file(s)")
        
        print("\n" + "="*80)
        print("IMPORTANT: This system ONLY uses the 3 data sources above.")
        print("It does NOT use: data/raw/, data/interventions/, data/reports/, or any other folders.")
        print("="*80 + "\n")
        
        # Initialize components
        self.ipc_parser = IPCParser(str(self.ipc_file))
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
    
    def setup_vector_stores(self, force_recreate: bool = False):
        """
        Setup separate vector stores for reports and interventions.
        
        ONLY uses:
        - Situation reports from: current situation report/
        - Intervention literature from: intervention-literature/
        
        Does NOT use: data/reports/, data/interventions/, or any other folders.
        """
        print("\n" + "="*80)
        print("SETTING UP VECTOR STORES")
        print("="*80)
        print("\n⚠️  ONLY processing from:")
        print(f"   - Situation reports: {self.reports_dir}")
        print(f"   - Intervention literature: {self.interventions_dir}")
        print("   - NOT using data/reports/ or data/interventions/\n")
        
        # Check if vector stores already exist
        reports_db_path = Path("./chroma_db_reports")
        interventions_db_path = Path("./chroma_db_interventions")
        
        if not force_recreate and reports_db_path.exists() and list(reports_db_path.glob("*.sqlite3")):
            print(f"\n✅ Reports vector store already exists at {reports_db_path}")
            print("   Loading existing vector store...")
            try:
                self.reports_vectorstore = Chroma(
                    persist_directory=str(reports_db_path),
                    embedding_function=self.embeddings
                )
                count = self.reports_vectorstore._collection.count()
                if count == 0:
                    print(f"   ⚠️  WARNING: Vector store exists but is EMPTY ({count} chunks)")
                    print(f"   Will regenerate from situation report PDFs...")
                    force_recreate = True
                    self.reports_vectorstore = None
                else:
                    print(f"   ✅ Loaded {count} chunks (skipping regeneration)")
            except Exception as e:
                print(f"   ⚠️  Error loading, will recreate: {e}")
                force_recreate = True
        
        if not force_recreate and interventions_db_path.exists() and list(interventions_db_path.glob("*.sqlite3")):
            print(f"\n✅ Interventions vector store already exists at {interventions_db_path}")
            print("   Loading existing vector store...")
            try:
                self.interventions_vectorstore = Chroma(
                    persist_directory=str(interventions_db_path),
                    embedding_function=self.embeddings
                )
                count = self.interventions_vectorstore._collection.count()
                if count == 0:
                    print(f"   ⚠️  WARNING: Vector store exists but is EMPTY ({count} chunks)")
                    print(f"   Will regenerate from intervention PDFs...")
                    force_recreate = True
                    self.interventions_vectorstore = None
                else:
                    print(f"   ✅ Loaded {count} chunks (skipping regeneration)")
            except Exception as e:
                print(f"   ⚠️  Error loading, will recreate: {e}")
                force_recreate = True
        
        # Process situation reports (only if needed)
        if force_recreate or self.reports_vectorstore is None:
            if self.reports_dir.exists():
                print(f"\n📄 Processing situation reports from: {self.reports_dir}")
                report_files = list(self.reports_dir.glob("*.pdf"))
                if report_files:
                    print(f"   Found {len(report_files)} PDF file(s)")
                    report_chunks = []
                    for pdf_file in report_files:
                        print(f"   Processing: {pdf_file.name}...")
                        chunks = self.doc_processor.process_pdf(str(pdf_file))
                        report_chunks.extend(chunks)
                        print(f"      Extracted {len(chunks)} chunks")
                    
                    if report_chunks:
                        print(f"\n   Creating documents from {len(report_chunks)} chunks...")
                        documents = [
                            Document(
                                page_content=chunk["content"],
                                metadata={"source": chunk["source"], "type": "situation_report"}
                            )
                            for chunk in report_chunks
                        ]
                        
                        print(f"   Generating embeddings for {len(documents)} chunks...")
                        print("   ⏳ This may take several minutes (no progress indicator available)...")
                        self.reports_vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=self.embeddings,
                            persist_directory="./chroma_db_reports"
                        )
                        
                        # Verify the vector store was created correctly
                        try:
                            actual_count = self.reports_vectorstore._collection.count()
                            if actual_count != len(documents):
                                print(f"   ⚠️  WARNING: Expected {len(documents)} chunks, but vector store has {actual_count}")
                            else:
                                print(f"   ✅ Created reports vector store with {actual_count} chunks")
                        except Exception as e:
                            print(f"   ⚠️  Could not verify chunk count: {e}")
                            print(f"   ✅ Created reports vector store (verification failed)")
                    else:
                        print("   ⚠️  No content extracted from reports")
                else:
                    print(f"   ⚠️  No PDF files found in {self.reports_dir}")
            else:
                print(f"   ⚠️  Reports directory not found: {self.reports_dir}")
        
        # Process intervention literature (only if needed)
        if force_recreate or self.interventions_vectorstore is None:
            if self.interventions_dir.exists():
                print(f"\n📚 Processing intervention literature from: {self.interventions_dir}")
                intervention_files = list(self.interventions_dir.glob("*.pdf"))
                if intervention_files:
                    print(f"   Found {len(intervention_files)} PDF file(s)")
                    
                    # Check for large PDFs and warn
                    large_pdfs = []
                    for pdf_file in intervention_files:
                        try:
                            reader = PdfReader(str(pdf_file))
                            page_count = len(reader.pages)
                            if page_count > 200:  # Warn if PDF has more than 200 pages
                                large_pdfs.append((pdf_file.name, page_count))
                        except:
                            pass
                    
                    if large_pdfs:
                        for name, pages in large_pdfs:
                            estimated_chunks = pages * 5  # Rough estimate: ~5 chunks per page
                            estimated_minutes = estimated_chunks // 60
                            print(f"   ⚠️  WARNING: {name} has {pages} pages.")
                            print(f"      This will create ~{estimated_chunks} chunks and take ~{estimated_minutes} minutes to process.")
                            print(f"      Consider excluding it or processing separately if needed.")
                    
                    intervention_chunks = []
                    for pdf_file in intervention_files:
                        print(f"   Processing: {pdf_file.name}...")
                        chunks = self.doc_processor.process_pdf(str(pdf_file))
                        intervention_chunks.extend(chunks)
                        print(f"      Extracted {len(chunks)} chunks")
                    
                    if intervention_chunks:
                        print(f"\n   Creating documents from {len(intervention_chunks)} chunks...")
                        documents = [
                            Document(
                                page_content=chunk["content"],
                                metadata={"source": chunk["source"], "type": "intervention_literature"}
                            )
                            for chunk in intervention_chunks
                        ]
                        
                        print(f"   Generating embeddings for {len(documents)} chunks...")
                        print(f"   ⏳ Estimated time: ~{len(documents)//60} minutes (no progress indicator available)...")
                        self.interventions_vectorstore = Chroma.from_documents(
                            documents=documents,
                            embedding=self.embeddings,
                            persist_directory="./chroma_db_interventions"
                        )
                        
                        # Verify the vector store was created correctly
                        try:
                            actual_count = self.interventions_vectorstore._collection.count()
                            if actual_count != len(documents):
                                print(f"   ⚠️  WARNING: Expected {len(documents)} chunks, but vector store has {actual_count}")
                            else:
                                print(f"   ✅ Created interventions vector store with {actual_count} chunks")
                        except Exception as e:
                            print(f"   ⚠️  Could not verify chunk count: {e}")
                            print(f"   ✅ Created interventions vector store (verification failed)")
                    else:
                        print("   ⚠️  No content extracted from intervention literature")
                else:
                    print(f"   ⚠️  No PDF files found in {self.interventions_dir}")
            else:
                print(f"   ⚠️  Interventions directory not found: {self.interventions_dir}")
    
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
        
        # Build expanded query with region context
        geographic_parts = assessment.geographic_full_name.split(',')
        region_variations = [region]
        for part in geographic_parts:
            part = part.strip()
            if part and part != region and part != "Ethiopia":
                region_variations.append(part)
        
        # Add zone/area context
        if "Burji" in region or "SNNPR" in assessment.geographic_full_name:
            region_variations.extend(["SNNPR", "Southern Nations", "southern Ethiopia", "agropastoral"])
        if "Tigray" in assessment.geographic_full_name:
            region_variations.extend(["Tigray", "northern Ethiopia"])
        if "Afar" in assessment.geographic_full_name:
            region_variations.extend(["Afar", "pastoral", "northeastern Ethiopia"])
        
        query = f"""
        Food security situation: {', '.join(region_variations[:5])}, Ethiopia.
        Current IPC phase: {assessment.current_phase}
        Livelihood system, shocks, drivers of food insecurity.
        Look for: conflict, drought, rainfall anomalies, price increases, displacement, 
        market disruptions, crop failures, livestock losses, livelihood impacts.
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
            # Retrieve more chunks for better context (increased from 5 to 10)
            retriever = self.reports_vectorstore.as_retriever(search_kwargs={"k": 10})
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
            
            # Build context from retrieved documents (increased chunk size for better context)
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:800]}"
                for doc in docs
            ])
            
            # Use LLM to extract drivers with new IPC-aligned prompt
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "context"],
                template="""You are a senior Integrated Food Security Phase Classification (IPC) analyst. 
Your task is to explain WHY {region} is experiencing IPC Phase {ipc_phase} food insecurity.

Use ONLY the information inside the situation report, but apply IPC-compatible 
regional and livelihood inference when the woreda is not directly mentioned.

===========================================================
MANDATORY ANALYSIS FRAMEWORK (MUST FOLLOW THIS ORDER)
===========================================================

1. LIVELIHOOD SYSTEM IDENTIFICATION
   - Identify the likely livelihood system (pastoral, agropastoral, cropping)
     using regional patterns, adjacent zones, or contextual clues. 
   - Explain why this livelihood system matters for food access.

2. SHOCK IDENTIFICATION (WHAT HAPPENED?)
   Extract all relevant shocks from the context, including:
   - rainfall anomalies / drought / flooding
   - conflict and insecurity
   - displacement
   - market disruptions and price spikes
   - crop failure
   - livestock disease or mortality
   - macroeconomic shocks
   - humanitarian access constraints

3. LIVELIHOOD IMPACT (HOW SHOCKS AFFECT FOOD/INCOME)
   For each shock, explain:
   - impact on agricultural production
   - impact on livestock body condition, births, milk, sales
   - impact on labor markets and wages
   - impact on market access, transport, supply chains
   - changes in terms of trade and purchasing power

4. FOOD ACCESS & CONSUMPTION GAPS
   - Describe how the above impacts reduce the ability to access food.
   - Identify reduced meal frequency, reduced dietary diversity, reliance on
     negative coping, distress sales, early depletion of stocks, etc.

5. NUTRITIONAL AND HEALTH OUTCOMES
   - Identify references to increasing GAM/SAM, disease outbreaks, water stress,
     or other factors increasing malnutrition.

6. IPC PHASE ALIGNMENT
   - Explicitly link the above evidence to IPC Phase {ipc_phase} outcomes:
     Phase 3 → Crisis level: consumption gaps, livelihood protection deficits
     Phase 4 → Emergency: extreme food deficits, acute malnutrition, asset collapse
     Phase 5 → Catastrophe/Famine: near-complete food consumption failure

7. LIMITATIONS
   - If the context does not name {region}, clearly state:
     "{region} is not directly mentioned; analysis is based on regional livelihood
      profiles and contextual patterns in the report."

===========================================================
FORMAT YOUR OUTPUT AS:
A. Overview
B. Livelihood System
C. Shocks
D. Livelihood Impacts
E. Food Access and Consumption
F. Nutrition & Health
G. IPC Alignment
H. Limitations
===========================================================

CONTEXT FROM SITUATION REPORTS:
{context}

Produce a detailed, structured IPC-style analytical narrative.
"""
            )
            
            chain = prompt | self.llm
            explanation = chain.invoke({
                "region": region,
                "ipc_phase": assessment.current_phase,
                "context": context
            })
            
            # Extract drivers (improved extraction based on structured output)
            drivers = []
            explanation_lower = explanation.lower()
            
            # Map to IPC driver categories
            if "conflict" in explanation_lower or "violence" in explanation_lower or "insecurity" in explanation_lower:
                drivers.append("Conflict and insecurity")
            if "drought" in explanation_lower or "rainfall" in explanation_lower or "dry" in explanation_lower or "deficit" in explanation_lower:
                drivers.append("Drought/Rainfall deficit")
            if "price" in explanation_lower or "cost" in explanation_lower or "market" in explanation_lower:
                drivers.append("Market disruptions and price increases")
            if "displacement" in explanation_lower or "displaced" in explanation_lower or "idp" in explanation_lower:
                drivers.append("Displacement")
            if "crop" in explanation_lower or "harvest" in explanation_lower or "agricultural" in explanation_lower:
                drivers.append("Crop failure")
            if "livestock" in explanation_lower or "pastoral" in explanation_lower:
                drivers.append("Livestock losses")
            if "flood" in explanation_lower or "flooding" in explanation_lower:
                drivers.append("Flooding")
            if "access" in explanation_lower and ("humanitarian" in explanation_lower or "constraint" in explanation_lower):
                drivers.append("Humanitarian access constraints")
            
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
        
        # Build query with generic intervention keywords to ensure retrieval
        # Intervention literature doesn't contain region names or IPC phase numbers,
        # so we need to search by intervention types and drivers
        intervention_keywords = [
            "emergency", "food security", "nutrition", "WFP", "Sphere",
            "CMAM", "SAM", "MAM", "cash", "voucher", "CVA", "LEGS",
            "livestock", "drought", "famine", "IPC", "response", "intervention",
            "therapeutic feeding", "food assistance", "water", "sanitation", "WASH"
        ]
        
        # Add driver-specific keywords
        driver_keywords = []
        if drivers:
            for driver in drivers:
                driver_lower = driver.lower()
                if "conflict" in driver_lower or "insecurity" in driver_lower:
                    driver_keywords.extend(["conflict", "protection", "displacement", "IDP"])
                if "drought" in driver_lower or "rainfall" in driver_lower:
                    driver_keywords.extend(["drought", "water", "livestock", "pastoral"])
                if "price" in driver_lower or "market" in driver_lower:
                    driver_keywords.extend(["cash", "voucher", "market", "price"])
                if "livestock" in driver_lower:
                    driver_keywords.extend(["livestock", "LEGS", "pastoral", "veterinary"])
                if "crop" in driver_lower:
                    driver_keywords.extend(["agricultural", "seeds", "crops", "harvest"])
                if "displacement" in driver_lower:
                    driver_keywords.extend(["displacement", "IDP", "shelter", "protection"])
        
        # Combine all keywords
        all_keywords = " ".join(intervention_keywords + driver_keywords)
        
        # Build query - prioritize generic intervention terms over region-specific terms
        query = f"""
        Emergency food security and nutrition interventions.
        IPC Phase {ipc_phase} ({'Emergency' if ipc_phase >= 4 else 'Crisis' if ipc_phase >= 3 else 'Stressed'}).
        Drivers: {drivers_str}.
        {all_keywords}
        Practical recommendations for small NGOs and individual aid workers.
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
            # Check if vector store has any documents
            try:
                doc_count = self.interventions_vectorstore._collection.count()
                if doc_count == 0:
                    print(f"   ⚠️  WARNING: Intervention vector store is empty ({doc_count} chunks)")
                    print(f"   Please regenerate vector stores: rm -rf chroma_db_interventions/ && restart")
                    return {
                        "region": region,
                        "recommendations": (
                            f"For {region} with IPC Phase {ipc_phase}, the intervention vector store "
                            f"is empty. Please ensure intervention PDFs are processed and vector stores "
                            f"are regenerated."
                        ),
                        "sources": [],
                        "limitations": f"Vector store empty ({doc_count} chunks)"
                    }
            except:
                pass  # If count fails, try retrieval anyway
            
            # Retrieve more chunks for better context (increased from 5 to 10)
            retriever = self.interventions_vectorstore.as_retriever(search_kwargs={"k": 10})
            docs = retriever.get_relevant_documents(query)
            
            print(f"   🔍 Retrieved {len(docs)} documents from intervention literature")
            if len(docs) > 0:
                print(f"   Sources: {[d.metadata.get('source', 'Unknown') for d in docs[:3]]}")
            
            if not docs or len(docs) == 0:
                # Try a more generic query as fallback
                print("   ⚠️  No documents with specific query, trying generic fallback...")
                fallback_query = "emergency food security nutrition intervention Sphere WFP CMAM LEGS"
                docs = retriever.get_relevant_documents(fallback_query)
                print(f"   🔍 Fallback query retrieved {len(docs)} documents")
                
                if not docs or len(docs) == 0:
                    return {
                        "region": region,
                        "recommendations": (
                            f"For {region} with IPC Phase {ipc_phase}, I could not retrieve "
                            f"any documents from the intervention literature. The vector store may be "
                            f"empty or the query is not matching any content. Please check that "
                            f"intervention PDFs were processed correctly."
                        ),
                        "sources": [],
                        "limitations": "No documents retrieved from vector store"
                    }
            
            # Build context (increased chunk size for better context)
            context = "\n\n".join([
                f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:800]}"
                for doc in docs
            ])
            
            # Use LLM to generate recommendations with new driver-linked prompt
            ipc_phase_desc = 'Emergency' if ipc_phase >= 4 else 'Crisis' if ipc_phase >= 3 else 'Stressed'
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "ipc_phase_desc", "drivers", "context"],
                template="""You are a humanitarian emergency response advisor. 
Your job is to recommend interventions for {region} experiencing IPC Phase {ipc_phase}.

Use ONLY the intervention literature provided in the context.

===========================================================
MANDATORY STRUCTURE
===========================================================

1. LINK INTERVENTIONS TO DRIVERS
   For each driver listed, map the appropriate intervention category using:
   - Sphere Minimum Standards
   - Cash & Voucher (CVA) guidance (CALP, UNHCR/WFP)
   - WFP Essential Needs, GFD, and EFSA guidance
   - Nutrition in Emergencies protocols (CMAM, IYCF-E, SAM/MAM treatment)
   - Livestock Emergency Guidelines (LEGS)
   - FAO agricultural support
   - Cluster (FSC/GNC) guidance
   - USAID/BHA emergency guidelines

2. IPC PHASE–SPECIFIC PRIORITIZATION
   For IPC Phase 3 (Crisis):
     - Prioritize livelihood protection, market support, early cash, agricultural inputs.
   For IPC Phase 4 (Emergency):
     - Prioritize lifesaving food assistance, SAM treatment, water trucking, survival livestock support.
   For IPC Phase 5 (Catastrophe/Famine):
     - Prioritize blanket food distributions, therapeutic feeding, emergency water, any life-saving measures.

3. INTERVENTION CATEGORIES
   Break interventions into:
   - Immediate life-saving actions
   - Food assistance and nutrition support
   - Livelihood protection (livestock feed/vet care, seed support)
   - Cash/voucher programming (CVA)
   - Market and price stabilization measures
   - WASH essential interventions
   - Protection and displacement considerations
   - Medium-term resilience and recovery

4. BEST PRACTICE REFERENCES
   Cite the intervention literature by name or theme:
   - "Sphere Handbook: Food Security & Nutrition"
   - "WFP Essential Needs Guidelines"
   - "LEGS Livestock Standards"
   - "CMAM/WHO SAM treatment protocols"
   - "CALP Cash & Voucher Assistance"

5. LIMITATIONS
   - If the context lacks direct instructions for a driver, acknowledge the gap.

===========================================================
FORMAT OUTPUT AS:
A. Summary
B. Immediate Emergency Actions
C. Food & Nutrition Interventions
D. Cash and Market Support
E. Livelihood Protection
F. WASH and Health Linkages
G. Coordination Requirements
H. Medium-term Recovery
I. Limitations
===========================================================

REGION: {region}
IPC PHASE: {ipc_phase} ({ipc_phase_desc})
DRIVERS: {drivers}

INTERVENTION LITERATURE:
{context}

All recommendations MUST be evidence-based and tied to the literature.
"""
            )
            
            chain = prompt | self.llm
            recommendations = chain.invoke({
                "region": region,
                "ipc_phase": ipc_phase,
                "ipc_phase_desc": ipc_phase_desc,
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

