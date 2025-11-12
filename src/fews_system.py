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
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from pypdf import PdfReader

from .ipc_parser import IPCParser, RegionRiskAssessment
from .document_processor import DocumentProcessor
from .domain_knowledge import DomainKnowledge
from .config import (
    MIN_RELEVANT_CHUNKS, MAX_CONTEXT_CHUNKS, CHUNK_SIZE,
    SHOCK_CONFIDENCE_THRESHOLD, MAX_SHOCKS_TO_RETURN
)
from .exceptions import (
    InsufficientDataError, VectorStoreError, RetrievalError,
    DomainKnowledgeError
)


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
        self.domain_knowledge = DomainKnowledge()
        
        # Vector stores
        self.reports_vectorstore: Optional[Chroma] = None
        self.interventions_vectorstore: Optional[Chroma] = None
        
        # LLM and embeddings
        self.embeddings = None
        self.llm = None
        
        # Initialize
        self._initialize()
    
    def _initialize(self) -> None:
        """Initialize embeddings and LLM."""
        try:
            self.embeddings = OllamaEmbeddings(model=self.model_name)
            self.llm = OllamaLLM(model=self.model_name)
            print("✅ Initialized LLM and embeddings")
        except Exception as e:
            print(f"⚠️  Error initializing LLM: {e}")
            print("   Make sure Ollama is running: ollama serve")
            print(f"   And model is available: ollama pull {self.model_name}")
    
    def _retrieve_context(
        self,
        query: str,
        vectorstore: Optional[Chroma],
        k: int = MAX_CONTEXT_CHUNKS,
        min_chunks: int = MIN_RELEVANT_CHUNKS
    ) -> Tuple[List[Document], str]:
        """
        Unified context retrieval with deduplication and validation.
        
        Args:
            query: Search query
            vectorstore: Chroma vector store to search
            k: Maximum number of chunks to retrieve
            min_chunks: Minimum chunks required for sufficiency
        
        Returns:
            Tuple of (documents, context_string)
        
        Raises:
            VectorStoreError: If vector store is None
            RetrievalError: If retrieval fails
            InsufficientDataError: If fewer than min_chunks retrieved
        """
        if vectorstore is None:
            raise VectorStoreError("Vector store is not initialized")
        
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.get_relevant_documents(query)
        except Exception as e:
            raise RetrievalError(f"Failed to retrieve documents: {e}")
        
        # Check minimum chunks requirement
        if len(docs) < min_chunks:
            raise InsufficientDataError(
                f"Insufficient context retrieved: {len(docs)} chunks (minimum {min_chunks} required)"
            )
        
        # Deduplicate by content hash (first 100 chars as hash)
        seen = set()
        unique_docs = []
        for doc in docs:
            content_hash = hash(doc.page_content[:100])
            if content_hash not in seen:
                seen.add(content_hash)
                unique_docs.append(doc)
        
        # Build context (normalized chunk size)
        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:CHUNK_SIZE]}"
            for doc in unique_docs
        ])
        
        return unique_docs, context
    
    def _detect_validated_shocks(
        self,
        context: str,
        region: str,
        assessment: RegionRiskAssessment
    ) -> List[str]:
        """
        Detect shocks using structured keyword matching.
        Returns ONLY shocks found in shock_ontology.json.
        
        Args:
            context: Retrieved text context
            region: Region name
            assessment: IPC risk assessment
        
        Returns:
            List of validated shock driver names
        """
        # Run keyword detection on context
        detected = self.domain_knowledge.detect_shocks(context)
        
        # Filter by confidence threshold
        validated_shocks = [
            shock_type for shock_type, confidence in detected
            if confidence >= SHOCK_CONFIDENCE_THRESHOLD
        ][:MAX_SHOCKS_TO_RETURN]
        
        # Map to driver names
        shock_to_driver = {
            "drought": "Drought/Rainfall deficit",
            "conflict": "Conflict and insecurity",
            "displacement": "Displacement",
            "price_increase": "Price increases",
            "crop_pests": "Crop failure",
            "livestock_mortality": "Livestock losses",
            "market_disruption": "Market disruption",
            "humanitarian_access_constraints": "Humanitarian access constraints",
            "flooding": "Flooding",
            "macroeconomic_shocks": "Macroeconomic shocks"
        }
        
        drivers = [
            shock_to_driver.get(s, s)
            for s in validated_shocks
            if s in shock_to_driver
        ]
        
        return drivers
    
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
        
        # Get structured domain knowledge
        geographic_parts = assessment.geographic_full_name.split(',')
        region_name = geographic_parts[0].strip() if geographic_parts else region
        zone_name = geographic_parts[1].strip() if len(geographic_parts) > 1 else None
        
        # Get livelihood system from domain knowledge
        livelihood_info = self.domain_knowledge.get_livelihood_system(region_name, zone_name)
        livelihood_system = livelihood_info.livelihood_system if livelihood_info else None
        
        # Get rainfall season from domain knowledge
        rainfall_info = self.domain_knowledge.get_rainfall_season(region_name, zone_name)
        dominant_season = rainfall_info.dominant_season if rainfall_info else None
        
        # Build expanded query with region context
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
            # Use unified retrieval function with sufficiency check
            try:
                docs, context = self._retrieve_context(
                    query=query,
                    vectorstore=self.reports_vectorstore,
                    k=MAX_CONTEXT_CHUNKS,
                    min_chunks=MIN_RELEVANT_CHUNKS
                )
            except InsufficientDataError as e:
                explanation = (
                    f"Region {region} has IPC Phase {assessment.current_phase}, "
                    f"indicating {'Emergency' if assessment.current_phase >= 4 else 'Crisis' if assessment.current_phase >= 3 else 'Stressed'} conditions. "
                    f"However, insufficient context was retrieved from situation reports "
                    f"({str(e)}). Cannot produce an evidence-based explanation."
                )
                missing_info_logger.warning(
                    f"Region: {region} | IPC Phase: {assessment.current_phase} | "
                    f"Issue: {str(e)}"
                )
                return {
                    "region": region,
                    "explanation": explanation,
                    "drivers": [],
                    "sources": [],
                    "data_quality": "insufficient",
                    "ipc_phase": assessment.current_phase
                }
            except (VectorStoreError, RetrievalError) as e:
                explanation = f"Error accessing situation reports: {str(e)}"
                missing_info_logger.error(f"Region: {region} | Error: {str(e)}")
                return {
                    "region": region,
                    "explanation": explanation,
                    "drivers": [],
                    "sources": [],
                    "data_quality": "error",
                    "ipc_phase": assessment.current_phase
                }
            
            # Detect validated shocks BEFORE prompting LLM
            validated_drivers = self._detect_validated_shocks(context, region, assessment)
            
            # Get shock types for domain context
            detected_shocks = self.domain_knowledge.detect_shocks(context)
            shock_types = [shock[0] for shock in detected_shocks[:MAX_SHOCKS_TO_RETURN]]
            
            # Build domain knowledge context for prompt (MANDATORY - LLM cannot infer these)
            domain_context = ""
            if livelihood_info:
                domain_context += f"\nLIVELIHOOD SYSTEM (MANDATORY - from domain knowledge CSV): {livelihood_info.livelihood_system} ({livelihood_info.elevation_category})\n"
                domain_context += f"  → You MUST use this livelihood system. Do NOT infer or guess a different one.\n"
            else:
                domain_context += f"\nLIVELIHOOD SYSTEM: Not found in domain knowledge. Use 'Unknown livelihood system'.\n"
                domain_context += f"  → Do NOT infer or guess a livelihood system.\n"
            
            if rainfall_info:
                domain_context += f"\nRAINFALL SEASON (MANDATORY - from domain knowledge CSV): {rainfall_info.dominant_season} ({rainfall_info.season_months})\n"
                domain_context += f"  → You MUST use this season name when discussing rainfall. Do NOT use other season names.\n"
            else:
                domain_context += f"\nRAINFALL SEASON: Not found in domain knowledge.\n"
                domain_context += f"  → Use neutral language: 'below-average rainfall' or 'rainfall anomalies'. Do NOT name specific seasons.\n"
            
            if validated_drivers:
                domain_context += f"\nVALIDATED SHOCKS (MANDATORY - from structured detection): {', '.join(validated_drivers)}\n"
                domain_context += f"  → You MUST only discuss these shocks. Do NOT add shocks not in this list.\n"
            else:
                domain_context += f"\nVALIDATED SHOCKS: None detected via structured keyword matching.\n"
                domain_context += f"  → If context mentions shocks, use only those explicitly stated. Do NOT infer additional shocks.\n"
            
            # Use LLM to extract drivers with strict IPC-aligned prompt
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "context", "domain_context", "validated_shocks"],
                template="""You are a senior Integrated Food Security Phase Classification (IPC) analyst. 
Your task is to explain WHY {region} is experiencing IPC Phase {ipc_phase} food insecurity.

CRITICAL CONSTRAINT: You MUST use ONLY the information provided in:
1. DOMAIN KNOWLEDGE CONTEXT (below) - these are MANDATORY lookups from CSV/JSON files
2. CONTEXT FROM SITUATION REPORTS (below) - retrieved text from vector database

You MUST NOT infer, guess, or invent:
- Livelihood systems (MUST use domain knowledge CSV lookup)
- Rainfall seasons (MUST use domain knowledge CSV lookup)
- Shocks (MUST use only validated shocks from structured detection)

===========================================================
CRITICAL CONSTRAINTS (MANDATORY)
===========================================================

1. LIVELIHOOD SYSTEM:
   - You MUST use the livelihood system specified in DOMAIN KNOWLEDGE CONTEXT.
   - If it says "Unknown livelihood system", state that explicitly.
   - DO NOT infer or guess a livelihood system.

2. RAINFALL SEASON:
   - You MUST use the rainfall season specified in DOMAIN KNOWLEDGE CONTEXT.
   - If it says to use neutral language, use "below-average rainfall" or "rainfall anomalies".
   - DO NOT name seasons (kiremt, deyr/hageya, gu/genna) unless explicitly provided.

3. SHOCKS:
   - You MUST only discuss shocks listed in VALIDATED SHOCKS.
   - DO NOT add shocks not in that list, even if context mentions them.
   - If validated shocks list is empty, state "No validated shocks detected via structured keyword matching."

===========================================================
MANDATORY ANALYSIS FRAMEWORK (MUST FOLLOW THIS EXACT ORDER)
===========================================================

A. Overview
   - Brief summary of the situation

B. Livelihood System
   - State the livelihood system from DOMAIN KNOWLEDGE CONTEXT
   - Explain why this system matters for food access
   - DO NOT infer a different livelihood system

C. Seasonal Calendar
   - State the rainfall season from DOMAIN KNOWLEDGE CONTEXT
   - If not provided, use neutral language without naming seasons
   - DO NOT infer season names

D. Shocks
   - List ONLY the validated shocks from VALIDATED SHOCKS
   - For each shock, cite evidence from CONTEXT FROM SITUATION REPORTS
   - DO NOT add shocks not in the validated list

E. Livelihood Impacts
   - Explain how shocks affect agricultural production, livestock, labor markets, market access
   - Base explanations on CONTEXT FROM SITUATION REPORTS

F. Food Access and Consumption
   - Describe consumption gaps, meal frequency, dietary diversity, coping strategies
   - Base on CONTEXT FROM SITUATION REPORTS

G. Nutrition & Health
   - Identify GAM/SAM, disease outbreaks, water stress if mentioned in context
   - If not mentioned, state "No specific nutrition/health data found"

H. IPC Alignment
   - Link evidence to IPC Phase {ipc_phase} outcomes
   - Phase 3 → Crisis: consumption gaps, livelihood protection deficits
   - Phase 4 → Emergency: extreme food deficits, acute malnutrition, asset collapse
   - Phase 5 → Catastrophe/Famine: near-complete food consumption failure

I. Limitations
   - State if {region} is not directly mentioned in context
   - State if domain knowledge is missing
   - State if validated shocks list is empty
   - DO NOT infer missing information

===========================================================
DOMAIN KNOWLEDGE CONTEXT (MANDATORY - DO NOT DEVIATE):
{domain_context}

VALIDATED SHOCKS (MANDATORY - ONLY DISCUSS THESE):
{validated_shocks}

CONTEXT FROM SITUATION REPORTS:
{context}

Produce a detailed, structured IPC-style analytical narrative following the exact format above.
"""
            )
            
            chain = prompt | self.llm
            validated_shocks_str = ", ".join(validated_drivers) if validated_drivers else "None detected via structured keyword matching"
            
            explanation = chain.invoke({
                "region": region,
                "ipc_phase": assessment.current_phase,
                "domain_context": domain_context,
                "validated_shocks": validated_shocks_str,
                "context": context
            })
            
            # Use validated drivers (already detected before prompting)
            explanation_lower = explanation.lower()
            drivers = validated_drivers if validated_drivers else []
            
            # Fallback: if no validated shocks, try keyword matching on explanation
            if not drivers:
                if "conflict" in explanation_lower or "violence" in explanation_lower or "insecurity" in explanation_lower:
                    drivers.append("Conflict and insecurity")
                if "drought" in explanation_lower or "rainfall" in explanation_lower or "dry" in explanation_lower or "deficit" in explanation_lower:
                    drivers.append("Drought/Rainfall deficit")
                if "price" in explanation_lower or "cost" in explanation_lower or "market" in explanation_lower:
                    drivers.append("Price increases")
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
            # Use unified retrieval function with sufficiency check
            try:
                docs, context = self._retrieve_context(
                    query=query,
                    vectorstore=self.interventions_vectorstore,
                    k=MAX_CONTEXT_CHUNKS,
                    min_chunks=MIN_RELEVANT_CHUNKS
                )
            except InsufficientDataError as e:
                missing_info_logger.warning(
                    f"Region: {region} | IPC Phase: {ipc_phase} | "
                    f"Issue: {str(e)}"
                )
                return {
                    "region": region,
                    "recommendations": (
                        f"For {region} with IPC Phase {ipc_phase}, insufficient context was retrieved "
                        f"from intervention literature ({str(e)}). Cannot produce evidence-based recommendations."
                    ),
                    "sources": [],
                    "limitations": str(e)
                }
            except (VectorStoreError, RetrievalError) as e:
                missing_info_logger.error(f"Region: {region} | Error: {str(e)}")
                return {
                    "region": region,
                    "recommendations": f"Error accessing intervention literature: {str(e)}",
                    "sources": [],
                    "limitations": str(e)
                }
            
            # Log retrieval success
            unique_sources = list(set([doc.metadata.get('source', 'Unknown') for doc in docs]))
            print(f"   🔍 Retrieved {len(docs)} chunks from {len(unique_sources)} document(s)")
            if unique_sources:
                print(f"   Sources: {unique_sources[:3]}")
            
            # Get structured intervention mappings from domain knowledge (MANDATORY)
            intervention_mappings = {}
            missing_mappings = []
            
            if drivers:
                driver_to_shock = {
                    "Drought/Rainfall deficit": "drought",
                    "Conflict and insecurity": "conflict",
                    "Displacement": "displacement",
                    "Price increases": "price_increase",
                    "Crop failure": "crop_pests",
                    "Livestock losses": "livestock_mortality",
                    "Market disruption": "market_disruption",
                    "Humanitarian access constraints": "humanitarian_access_constraints",
                    "Flooding": "flooding",
                    "Macroeconomic shocks": "macroeconomic_shocks"
                }
                
                for driver in drivers:
                    shock_type = driver_to_shock.get(driver)
                    if shock_type:
                        interventions = self.domain_knowledge.get_interventions_for_driver(shock_type)
                        if interventions:
                            intervention_mappings[driver] = interventions
                        else:
                            missing_mappings.append(driver)
                            missing_info_logger.warning(
                                f"Region: {region} | Driver: {driver} | "
                                f"Issue: No intervention mapping found in domain knowledge"
                            )
                    else:
                        missing_mappings.append(driver)
                        missing_info_logger.warning(
                            f"Region: {region} | Driver: {driver} | "
                            f"Issue: Driver not mapped to shock type"
                        )
            
            # Build intervention context (MANDATORY - LLM must use these mappings)
            intervention_context = ""
            if intervention_mappings:
                intervention_context = "\nSTRUCTURED INTERVENTION MAPPINGS (MANDATORY - from domain knowledge JSON):\n"
                intervention_context += "You MUST use these mappings to guide your recommendations.\n\n"
                for driver, interventions in intervention_mappings.items():
                    intervention_context += f"\n{driver}:\n"
                    for category, items in interventions.items():
                        if category != "references" and items:
                            intervention_context += f"  {category}: {', '.join(items[:3])}\n"
                    if "references" in interventions:
                        intervention_context += f"  References: {', '.join(interventions['references'][:2])}\n"
            
            if missing_mappings:
                intervention_context += f"\n⚠️  MISSING MAPPINGS: The following drivers have no structured intervention mapping: {', '.join(missing_mappings)}\n"
                intervention_context += "  → Use general best-practice guidance from intervention literature for these drivers.\n"
            
            # Use LLM to generate recommendations with new driver-linked prompt
            ipc_phase_desc = 'Emergency' if ipc_phase >= 4 else 'Crisis' if ipc_phase >= 3 else 'Stressed'
            prompt = PromptTemplate(
                input_variables=["region", "ipc_phase", "ipc_phase_desc", "drivers", "context", "intervention_context"],
                template="""You are a humanitarian emergency response advisor. 
Your job is to recommend interventions for {region} experiencing IPC Phase {ipc_phase}.

Use ONLY the intervention literature provided in the context.

===========================================================
MANDATORY STRUCTURE
===========================================================

1. LINK INTERVENTIONS TO DRIVERS
   For each major driver in DRIVERS, map at least one relevant intervention or 
   package of interventions, drawing on:
   - Sphere Minimum Standards (Food Security & Nutrition, WASH)
   - Cash & Voucher (CVA) guidance (CALP, UNHCR/WFP)
   - WFP Essential Needs, GFD, and EFSA guidelines
   - Nutrition in Emergencies protocols (CMAM, IYCF-E, SAM/MAM treatment)
   - Livestock Emergency Guidelines and Standards (LEGS)
   - FAO agricultural and livelihood support
   - Food Security & Nutrition Cluster (FSC/GNC) guidance
   - USAID/BHA emergency and early recovery guidelines

2. IPC PHASE–SPECIFIC PRIORITIZATION
   For IPC Phase 3 (Crisis):
     - Prioritize livelihood protection, market support, early cash,
       agricultural inputs, and targeted food assistance.
   For IPC Phase 4 (Emergency):
     - Prioritize life-saving food assistance, SAM treatment, water trucking,
       survival livestock support, and high-intensity CVA where markets function.
   For IPC Phase 5 (Catastrophe/Famine):
     - Prioritize blanket food distributions, therapeutic feeding, emergency water,
       and any other life-saving measures, with less emphasis on medium-term recovery.

3. MANDATORY INTERVENTION COVERAGE
   You MUST consider and, where relevant, recommend interventions in each of the
   following domains (even if briefly):

   - FOOD:
     - General food distribution (GFD), vouchers, or cash-based transfers aligned
       with essential needs.
   - NUTRITION:
     - CMAM programming, SAM/MAM treatment, blanket supplementary feeding where needed,
       IYCF-E support for infants and young children.
   - LIVESTOCK:
     - LEGS-consistent support: emergency feed, water for livestock, veterinary care,
       strategic destocking, restocking where appropriate.
   - AGRICULTURE & LIVELIHOODS:
     - Seeds, tools, support for next season planting, small-scale irrigation, soil
       and water conservation, support to petty trade and small businesses.
   - CASH & MARKETS:
     - CVA (multi-purpose cash, vouchers), trader credit where appropriate, support
       to restore supply chains and stabilize markets.
   - WASH:
     - Water trucking, rehabilitation of water points, water quality treatment,
       hygiene promotion, WASH in nutrition and health facilities.
   - HEALTH:
     - Disease surveillance, ORS/zinc, malaria prevention where relevant, linkages
       between nutrition and health services.
   - PROTECTION & DISPLACEMENT:
     - Safeguarding, GBV risk mitigation, safe access to assistance, attention to
       displaced populations and host communities.

4. BEST PRACTICE REFERENCES
   Where possible, explicitly reference the relevant guidance family, e.g.:
   - "Sphere Handbook: Food Security & Nutrition standards"
   - "Sphere WASH standards"
   - "LEGS livestock emergency guidelines"
   - "WFP Essential Needs Guidelines"
   - "CMAM/WHO SAM treatment protocols"
   - "CALP Cash & Voucher Assistance guidance"
   - "Global Food Security Cluster coordination guidance"

5. LIMITATIONS
   - If the literature does not directly address a very specific local constraint,
     acknowledge this, but still provide the best-practice intervention package 
     based on the closest relevant guidance.
   - You MUST NOT say "no recommendation is made" or leave a domain blank if 
     there is general best-practice guidance in the literature.

===========================================================
FORMAT OUTPUT AS:
A. Summary
B. Immediate Emergency Actions
C. Food & Nutrition Interventions
D. Cash and Market Support
E. Livelihood Protection (incl. livestock and agriculture)
F. WASH and Health Linkages
G. Coordination Requirements
H. Medium-term Recovery
I. Limitations
===========================================================

REGION: {region}
IPC PHASE: {ipc_phase} ({ipc_phase_desc})
DRIVERS: {drivers}

{intervention_context}

INTERVENTION LITERATURE:
{context}

All recommendations MUST be evidence-based and explicitly tied to the literature,
and must be clearly linked back to the drivers of food insecurity.
Use the structured intervention mappings above as a guide, but ground all recommendations
in the intervention literature provided.
"""
            )
            
            chain = prompt | self.llm
            recommendations = chain.invoke({
                "region": region,
                "ipc_phase": ipc_phase,
                "ipc_phase_desc": ipc_phase_desc,
                "drivers": drivers_str,
                "context": context,
                "intervention_context": intervention_context
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

