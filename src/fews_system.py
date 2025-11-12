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
        
        # Retrieval configuration
        self.retrieval_k = 8
        
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
        k: Optional[int] = None,
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
        
        if k is None:
            k = self.retrieval_k
        
        try:
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(query)
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
    ) -> Tuple[List[str], List[str]]:
        """
        Detect shocks using structured keyword matching.
        Returns shocks and mapped drivers.
        
        Args:
            context: Retrieved text context
            region: Region name
            assessment: IPC risk assessment
        
        Returns:
            Tuple of (shock_types, driver_names)
        """
        # Run keyword detection on context
        detected = self.domain_knowledge.detect_shocks(context)
        
        # Filter by confidence threshold
        validated_shocks = [
            shock_type for shock_type, confidence in detected
            if confidence >= SHOCK_CONFIDENCE_THRESHOLD
        ][:MAX_SHOCKS_TO_RETURN]
        
        drivers = self.domain_knowledge.map_shocks_to_drivers(validated_shocks)
        
        return validated_shocks, drivers
    
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
        geographic_parts = [part.strip() for part in assessment.geographic_full_name.split(',') if part.strip()]
        admin_name = geographic_parts[0] if geographic_parts else region
        country_name = geographic_parts[-1] if geographic_parts else "Ethiopia"
        region_name = geographic_parts[-2] if len(geographic_parts) >= 2 else None
        zone_name = geographic_parts[-3] if len(geographic_parts) >= 3 else None
        
        if zone_name and zone_name == admin_name:
            zone_name = None
        if region_name and region_name in ("Ethiopia", country_name):
            region_name = None
        if zone_name and region_name and zone_name == region_name:
            zone_name = None
        
        lookup_region = region_name or zone_name or admin_name
        
        # Get livelihood system from domain knowledge
        livelihood_info = self.domain_knowledge.get_livelihood_system(
            region=lookup_region,
            zone=zone_name,
            admin=admin_name
        )
        if livelihood_info:
            livelihood_system_for_prompt = f"{livelihood_info.livelihood_system} ({livelihood_info.elevation_category})"
            livelihood_notes = livelihood_info.notes if isinstance(livelihood_info.notes, str) else ""
        else:
            livelihood_system_for_prompt = "No livelihood data available for this region."
            livelihood_notes = ""
        
        # Get rainfall season from domain knowledge
        rainfall_info = self.domain_knowledge.get_rainfall_season(
            region=lookup_region,
            zone=zone_name,
            admin=admin_name
        )
        if rainfall_info:
            if rainfall_info.secondary_season:
                rainfall_season_for_prompt = f"{rainfall_info.dominant_season} (Secondary: {rainfall_info.secondary_season})"
            else:
                rainfall_season_for_prompt = rainfall_info.dominant_season
            rainfall_notes = rainfall_info.notes if isinstance(rainfall_info.notes, str) else ""
            rainfall_months = rainfall_info.season_months if isinstance(rainfall_info.season_months, str) else ""
        else:
            rainfall_season_for_prompt = "No rainfall season data available for this region."
            rainfall_notes = ""
            rainfall_months = ""
        
        # Build expanded query with region context
        region_variations = []
        for name in [region, admin_name, zone_name, region_name]:
            if name and name not in region_variations:
                region_variations.append(name)
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
            validated_shock_types, validated_drivers = self._detect_validated_shocks(context, region, assessment)
            if not validated_shock_types:
                missing_info_logger.info(
                    f"Region: {region} | IPC Phase: {assessment.current_phase} | "
                    "Notice: No validated shocks detected via structured keyword matching."
                )
            
            # Build domain knowledge context for prompt (reference only)
            domain_context_parts: List[str] = []
            if livelihood_info:
                domain_context_parts.append(
                    f"LIVELIHOOD DETAILS: {livelihood_info.livelihood_system} "
                    f"({livelihood_info.elevation_category}). Notes: {livelihood_info.notes}"
                )
            if rainfall_info:
                rainfall_detail = f"{rainfall_info.dominant_season}"
                if rainfall_info.secondary_season:
                    rainfall_detail += f" | Secondary: {rainfall_info.secondary_season}"
                rainfall_detail += f". Months: {rainfall_info.season_months}. Notes: {rainfall_info.notes}"
                domain_context_parts.append(f"RAINFALL DETAILS: {rainfall_detail}")
            if validated_shock_types:
                domain_context_parts.append(f"VALIDATED SHOCK LABELS: {', '.join(validated_shock_types)}")
            if validated_drivers:
                domain_context_parts.append(f"VALIDATED DRIVER LABELS: {', '.join(validated_drivers)}")
            domain_context = "\n".join(domain_context_parts)
            
            # Use LLM to extract drivers with strict IPC-aligned prompt
            prompt = PromptTemplate(
                input_variables=[
                    "region",
                    "ipc_phase",
                    "context",
                    "domain_context",
                    "validated_shocks",
                    "livelihood_system",
                    "rainfall_season"
                ],
                template="""You are a senior Integrated Food Security Phase Classification (IPC) analyst. 
Your task is to explain WHY {region} is experiencing IPC Phase {ipc_phase} food insecurity.

You MUST anchor all reasoning to the structured inputs below. They override any inference from narrative text.

LIVELIHOOD SYSTEM: {livelihood_system}
RAINFALL SEASON: {rainfall_season}
IPC PHASE FOR THIS REGION: {ipc_phase}
VALIDATED SHOCKS FOR THIS REGION: {validated_shocks}

You MUST treat the above values as complete and authoritative:
- Use the livelihood system exactly as written. Do NOT say it is unknown or unclear unless it explicitly states "No livelihood data available for this region."
- Use the rainfall season exactly as written. Do NOT substitute different seasons. If it states "No rainfall season data available..." you may only reference rainfall generically.
- Treat the IPC phase as definitive. You may NOT claim the phase is unknown, misaligned, or cannot be determined.
- The validated shocks list is the full and final set of shocks you are allowed to discuss.
  * If the list is empty, you MUST say “No validated shocks detected.” and you MUST NOT speculate about other shocks.
  * If the list is not empty, you MUST list ONLY these shocks and MUST NOT say shocks are missing or unknown.

You MUST use ONLY:
1. DOMAIN KNOWLEDGE CONTEXT (below) - reference information to support explanations
2. CONTEXT FROM SITUATION REPORTS (below) - retrieved narrative evidence

===========================================================
MANDATORY ANALYSIS FRAMEWORK (MUST FOLLOW THIS EXACT ORDER)
===========================================================

A. Overview
   - Brief summary of the situation

B. Livelihood System
   - Restate the livelihood system from LIVELIHOOD SYSTEM above
   - Explain why this system matters for food access using contextual evidence

C. Seasonal Calendar
   - Restate the rainfall season from RAINFALL SEASON above
   - If rainfall season data is unavailable, use neutral rainfall language without naming seasons

D. Shocks
   - If VALIDATED SHOCKS is empty, state “No validated shocks detected.”
   - If not empty, list ONLY those shocks and cite evidence from CONTEXT FROM SITUATION REPORTS
   - NEVER add shocks that are not in the validated list

E. Livelihood Impacts
   - Describe impacts on production, livestock, labor, and markets ONLY if the retrieved context provides explicit evidence.
   - If no evidence exists, state: “No specific livelihood impact details were found in the retrieved reports.”

F. Food Access and Consumption
   - Discuss consumption gaps, meal frequency, dietary diversity, or coping strategies ONLY if explicitly mentioned in the context.
   - If not mentioned, state: “No specific information about consumption gaps or coping strategies was found in the retrieved reports.”

G. Nutrition & Health
   - Discuss GAM/SAM, malnutrition, or health outcomes ONLY if the context explicitly mentions them.
   - If not mentioned, state: “No specific nutrition or health information was found in the retrieved reports.”

H. IPC Alignment
   - Link available evidence to IPC Phase {ipc_phase} outcomes.
   - You MUST affirm that the region is IPC Phase {ipc_phase}; you may discuss evidence gaps but may NOT claim misalignment or uncertainty about the phase.

I. Limitations
   - Mention missing livelihood or rainfall data ONLY if the structured values above state “No ... data available.”
   - Mention missing shocks ONLY if VALIDATED SHOCKS is empty.
   - State if {region} is not directly named in the retrieved context.
   - NEVER contradict the provided livelihood system, rainfall season, IPC phase, or validated shocks.

===========================================================
DOMAIN KNOWLEDGE CONTEXT (reference only):
{domain_context}

CONTEXT FROM SITUATION REPORTS:
{context}

Produce a detailed, structured IPC-style analytical narrative following the exact format above.
"""
            )
            
            chain = prompt | self.llm
            if validated_shock_types:
                validated_shocks_str = ", ".join(validated_shock_types)
            else:
                validated_shocks_str = "None detected via structured keyword matching"
            
            explanation = chain.invoke({
                "region": region,
                "ipc_phase": assessment.current_phase,
                "domain_context": domain_context,
                "validated_shocks": validated_shocks_str,
                "context": context,
                "livelihood_system": livelihood_system_for_prompt,
                "rainfall_season": rainfall_season_for_prompt
            })
            
            # Use validated drivers (already detected before prompting)
            explanation_lower = explanation.lower()
            drivers = list(validated_drivers)
            
            # Check if explanation indicates insufficient data
            data_quality = "sufficient"
            if not validated_shock_types:
                data_quality = "insufficient_shock_evidence"
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
                "drivers": drivers if drivers else [],
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
                    k=self.retrieval_k,
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

