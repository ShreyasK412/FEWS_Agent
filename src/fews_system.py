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
        assessment: RegionRiskAssessment,
        livelihood_zone: str = "unknown"
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Detect shocks using zone-specific structured keyword matching.
        Returns shocks, mapped drivers, and detailed results.
        
        Args:
            context: Retrieved text context
            region: Region name
            assessment: IPC risk assessment
            livelihood_zone: Livelihood system for zone-specific detection
        
        Returns:
            Tuple of (shock_types, driver_names, detailed_results)
        """
        # Run zone-specific keyword detection
        shock_types, detailed_results = self.domain_knowledge.detect_shocks_by_zone(
            text=context,
            livelihood_zone=livelihood_zone,
            region=region,
            ipc_phase=assessment.current_phase
        )
        
        # Filter by confidence threshold
        validated_shocks = []
        validated_details = []
        for i, shock_type in enumerate(shock_types):
            detail = detailed_results[i]
            if detail['confidence'] in ['high', 'medium']:
                validated_shocks.append(shock_type)
                validated_details.append(detail)
        
        # Limit to top shocks
        validated_shocks = validated_shocks[:MAX_SHOCKS_TO_RETURN]
        validated_details = validated_details[:MAX_SHOCKS_TO_RETURN]
        
        # Map to drivers
        drivers = self.domain_knowledge.map_shocks_to_drivers(validated_shocks)
        
        return validated_shocks, drivers, validated_details
    
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
            livelihood_system_for_prompt = "None"
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
            rainfall_season_for_prompt = "None"
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
        
        # Multi-query RAG retrieval strategy for better coverage
        queries = []
        
        # Query 1: Geographic query with region hierarchy
        geographic_query = f"{', '.join(region_variations[:3])} Ethiopia food security"
        queries.append(("geographic", geographic_query))
        
        # Query 2: Livelihood-specific query
        if livelihood_info:
            livelihood_query = f"{livelihood_info.livelihood_system} {region_name} Ethiopia current conditions"
            queries.append(("livelihood", livelihood_query))
        
        # Query 3: Seasonal query
        if rainfall_info:
            seasonal_query = f"{rainfall_info.dominant_season} {region_name} Ethiopia 2024 2025"
            queries.append(("seasonal", seasonal_query))
        
        # Query 4: IPC phase-specific query
        phase_query = f"IPC Phase {assessment.current_phase} {region_name} Ethiopia"
        queries.append(("phase", phase_query))
        
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
            # Multi-query retrieval: retrieve from each query and deduplicate
            all_docs = []
            for query_type, query in queries:
                try:
                    # Direct retrieval without min_chunks enforcement
                    if self.reports_vectorstore:
                        retriever = self.reports_vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(query)
                        if docs:
                            all_docs.extend(docs)
                            print(f"   📥 Query '{query_type}': retrieved {len(docs)} chunks")
                except Exception as e:
                    # Continue with other queries even if one fails
                    print(f"   ⚠️  Query '{query_type}' failed: {str(e)[:50]}")
                    pass
            
            # Deduplicate by content hash (using more content for better uniqueness)
            seen_hashes = set()
            unique_docs = []
            for doc in all_docs:
                # Use first 300 chars instead of 100 to reduce false duplicates
                content_hash = hash(doc.page_content[:300] if len(doc.page_content) >= 300 else doc.page_content)
                if content_hash not in seen_hashes:
                    seen_hashes.add(content_hash)
                    unique_docs.append(doc)
            
            # Limit to top chunks and build context
            unique_docs = unique_docs[:MAX_CONTEXT_CHUNKS]
            context = "\n\n".join([doc.page_content for doc in unique_docs])
            docs = unique_docs
            
            # Check sufficiency
            if len(docs) < MIN_RELEVANT_CHUNKS:
                raise InsufficientDataError(
                    f"Retrieved only {len(docs)} chunks, minimum required: {MIN_RELEVANT_CHUNKS}"
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
        
        # Detect validated shocks BEFORE prompting LLM (zone-specific detection)
        livelihood_zone_for_detection = livelihood_info.livelihood_system if livelihood_info else "unknown"
        validated_shock_types, validated_drivers, shock_details = self._detect_validated_shocks(
                context, region, assessment, livelihood_zone_for_detection
        )
        if not validated_shock_types:
                missing_info_logger.info(
                    f"Region: {region} | IPC Phase: {assessment.current_phase} | "
                    f"Notice: No validated shocks detected via structured keyword matching (zone: {livelihood_zone_for_detection})."
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
                    "validated_shocks",
                    "livelihood_system",
                    "rainfall_season"
                ],
                template="""You are a senior IPC Analyst. Your job is to explain WHY {region} is experiencing IPC Phase {ipc_phase} food insecurity.

IMPORTANT — AUTHORITATIVE DOMAIN KNOWLEDGE VALUES:
You are ALWAYS given the following fields directly from structured domain knowledge:
- LIVELIHOOD SYSTEM: {livelihood_system}
- RAINFALL SEASON: {rainfall_season}
- VALIDATED SHOCKS: {validated_shocks}

These values are ALWAYS correct and MUST NOT be contradicted, replaced, ignored, or described as "unknown," "not mentioned," "unclear," or "not provided."  
These values DO NOT come from reports — they come from authoritative domain knowledge.

LIMITATION RULE:
In the Limitations section, you may ONLY state a limitation if the input value is literally NULL or empty (i.e., "None").  
If a field is present, you MUST NOT claim it is missing.

===========================================================
MANDATORY ANALYSIS STRUCTURE
===========================================================

A. Overview  
A concise 2–3 sentence summary of the food security situation.

B. Livelihood System  
Use EXACTLY the livelihood system passed in: {livelihood_system}.  
Explain why this livelihood is sensitive to shocks.

C. Seasonal Calendar  
Use EXACTLY the rainfall season passed in: {rainfall_season}.  
Explain how that season affects production and food access.

D. Shocks  
List ONLY the validated shocks: {validated_shocks}.  
Do NOT infer or hallucinate shocks not included in the list.

E. Livelihood Impacts  
For each validated shock, explain impacts on:
- production  
- livestock  
- labor markets  
- market access  
- household purchasing power  
Never hallucinate impacts unrelated to validated shocks.

F. Food Access & Consumption  
Explain the consequences: consumption gaps, coping, market stress, etc.  
Stay tied to validated shocks.

G. Nutrition & Health  
If no nutrition data is in retrieved context, say:  
"No nutrition information available in retrieved context."

H. IPC Alignment  
Explain how the validated shocks + impacts lead to IPC Phase {ipc_phase}.  
Use official IPC interpretation rules.

I. Limitations  
Only include TRUE limitations:
- If no report text pertains to {region}, state that.
- If validated_shocks is empty, say so.
- DO NOT say livelihood system or rainfall season is missing if values were supplied (i.e., not "None").

===========================================================
CONTEXT FROM SITUATION REPORTS:
{context}

Produce a structured, contradiction-free explanation.
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
        
        # Get livelihood information for zone-appropriate intervention filtering
        geographic_parts = assessment.geographic_full_name.split(',') if hasattr(assessment, 'geographic_full_name') else []
        region_name = geographic_parts[-2].strip() if len(geographic_parts) >= 2 else ""
        zone_name = geographic_parts[1].strip() if len(geographic_parts) >= 2 else ""
        admin_name = region
        
        livelihood_info = self.domain_knowledge.get_livelihood_system(
            region=region_name,
            zone=zone_name,
            admin=admin_name
        )
        
        # Determine intervention filters based on livelihood zone
        intervention_filters = []
        livelihood_note = ""
        
        if livelihood_info:
            livelihood_system = livelihood_info.livelihood_system.lower()
            
            if 'rainfed' in livelihood_system or 'cropping' in livelihood_system or 'highland' in livelihood_system:
                # Highland cropping zone - filter out pastoral interventions
                intervention_filters = [
                    "destocking", "livestock feed", "pastoral", "herd", "milk production",
                    "pasture", "vaccination campaigns for livestock", "livestock water",
                    "animal health", "veterinary mobile units", "fodder", "restocking",
                    "herd management", "rangeland", "grazing"
                ]
                livelihood_note = (
                    f"CRITICAL LIVELIHOOD CONSTRAINT: {region} is in a HIGHLAND CROPPING zone "
                    f"({livelihood_info.livelihood_system}). You MUST NOT recommend pastoral interventions "
                    f"such as: destocking, livestock feed, pastoral water points, veterinary campaigns, "
                    f"herd management, or rangeland management. Instead, focus on: seed distribution, "
                    f"fertilizer, crop protection, agricultural labor support, mechanized farming, "
                    f"irrigation for crops, post-harvest storage, and food processing."
                )
            
            elif 'pastoral' in livelihood_system and 'agro' not in livelihood_system:
                # Pastoral zone - filter out crop-based interventions
                intervention_filters = [
                    "seed distribution", "fertilizer", "crop production", "harvest support",
                    "agricultural labor", "mechanized farming", "planting", "crop protection",
                    "irrigation for crops", "post-harvest", "food processing"
                ]
                livelihood_note = (
                    f"CRITICAL LIVELIHOOD CONSTRAINT: {region} is in a PASTORAL zone "
                    f"({livelihood_info.livelihood_system}). You MUST NOT recommend crop-based interventions "
                    f"such as: seed distribution, fertilizer, crop production, harvest support, or irrigation "
                    f"for crops. Instead, focus on: livestock feed, veterinary care, water for livestock, "
                    f"strategic destocking, restocking where appropriate, pasture management, and livestock "
                    f"market support."
                )
            
            else:  # agropastoral or mixed
                intervention_filters = []
                livelihood_note = (
                    f"LIVELIHOOD CONTEXT: {region} is in an AGROPASTORAL/MIXED zone "
                    f"({livelihood_info.livelihood_system}). You may recommend both crop-based and "
                    f"livestock-based interventions as appropriate to the drivers and IPC phase."
                )
        else:
            livelihood_note = (
                f"LIVELIHOOD CONTEXT: Livelihood system for {region} is not specified in domain knowledge. "
                f"Recommend interventions based on drivers and IPC phase, but note this limitation."
            )
        
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
                input_variables=["region", "ipc_phase", "ipc_phase_desc", "drivers", "context", "intervention_context", "livelihood_note"],
                template="""You are a humanitarian emergency response advisor. 
Your job is to recommend interventions for {region}, which is experiencing IPC Phase {ipc_phase}.

{livelihood_note}

AUTHORITATIVE DATA YOU MUST OBEY:
- DRIVERS (validated shocks only): {drivers}
- INTERVENTION MAPPINGS from driver_interventions.json (authoritative)
- INTERVENTION LITERATURE: {context}

You MUST:
- Map each driver → its matched intervention domains (from driver_interventions.json)
- Use the retrieved literature to justify each major intervention category
- NEVER hallucinate missing data or say "insufficient literature" unless context is literally empty
- NEVER contradict given drivers
- STRICTLY FOLLOW the livelihood constraint above (do not recommend filtered intervention types)

===========================================================
MANDATORY STRUCTURE
===========================================================

A. Summary  
2–3 sentences summarizing key emergency needs.

B. Immediate Emergency Actions  
Driven by IPC Phase {ipc_phase} priorities:
- Life-saving food assistance  
- WASH emergency measures  
- Protection mainstreaming  
- Any driver-linked immediate measures  

C. Food & Nutrition  
Use:
- Sphere Handbook Nutrition Standards
- CMAM/SAM/MAM protocols
- Blanket Supplementary Feeding rules

D. Cash & Markets  
Use:
- CALP CVA guidance
- Trader credit
- Market functionality thresholds

E. Livelihood Protection
Use validated LEGS livestock actions and agricultural support.

F. WASH and Health Linkages
Use Sphere WASH standards and emergency health linkages.

G. Coordination Requirements
Cluster coordination, IPC-consistent decision-making, information sharing.

H. Medium-Term Recovery  
Driver-linked livelihood and resilience-building measures.

I. Limitations
You may ONLY include:
- "Intervention literature contained no relevant guidance" if context == empty.
- Do NOT invent limitations.
- Do NOT claim missing drivers or missing domain knowledge.

===========================================================
REGION: {region}
IPC PHASE: {ipc_phase} ({ipc_phase_desc})
DRIVERS: {drivers}

{intervention_context}

INTERVENTION LITERATURE:
{context}

Output must be grounded ONLY in:
- driver_interventions.json
- retrieved literature
- IPC phase rules

Never output hallucinated limitations.
"""
            )
            
            chain = prompt | self.llm
            recommendations = chain.invoke({
                "region": region,
                "ipc_phase": ipc_phase,
                "ipc_phase_desc": ipc_phase_desc,
                "drivers": drivers_str,
                "context": context,
                "intervention_context": intervention_context,
                "livelihood_note": livelihood_note
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

