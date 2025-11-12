# ✅ CURSOR SUPER-PROMPT FOR A FULL FEWS REFACTOR

**Paste this directly into Cursor → `cmd+K` → "Refactor this project"**

---

## **SYSTEM INSTRUCTION FOR CURSOR**

You are refactoring a full Python project called **FEWS (Famine Early Warning System)**.
You must perform a full audit and improvement of the codebase, without changing functionality but **fixing the accuracy, data flow, reliability, and reasoning bottlenecks**.

Below is the project description and architecture. Everything you need is already inside the project.

---

# 🔥 **WHAT IS WRONG: THE CORE PROBLEMS TO FIX**

Your main goals:

---

## **1. RAG retrieval for situation reports is weak → model invents shocks**

* Current situation report repository is too small
* Retrieval sometimes gets irrelevant or generic text
* LLM fills gaps with guesses

### **Required fix**

Implement a **strict retrieval sufficiency check** across the codebase:

```python
MIN_RELEVANT_CHUNKS = 3

if len(retrieved_docs) < MIN_RELEVANT_CHUNKS:
    return {
        "explanation": "Insufficient context to produce an evidence-based explanation.",
        "data_quality": "insufficient",
        "drivers": []
    }
```

LLM MUST NOT reason beyond retrieved text + validated domain knowledge.

---

## **2. "Explain Why" sometimes invents:**

* rainfall seasons
* shock types
* crop calendars
* livelihood systems

Even with domain knowledge, some outputs still hallucinate.

### **Required fix: hard enforcement**

Rewrite the "Explain Why" logic and prompt so that:

1. **Livelihood system** MUST come from:
   `data/domain/livelihood_systems.csv` via `DomainKnowledge.get_livelihood_system()`
   - If not found → use fallback "Unknown livelihood system"
   - LLM cannot infer or guess

2. **Rainfall season** MUST come from:
   `data/domain/rainfall_seasons.csv` via `DomainKnowledge.get_rainfall_season()`
   - If not found → use neutral language "seasonal rainfall" without naming seasons
   - LLM cannot infer or guess

3. **Shock list** MUST come from domain shock detection:
   `data/domain/shock_ontology.json` via `DomainKnowledge.detect_shocks()`
   - LLM cannot add shocks not in this list
   - Only shocks detected via keyword matching are valid

4. **Anything not in vector store + not in domain knowledge must be explicitly disallowed.**

Add to prompt:
```
CRITICAL CONSTRAINT: You MUST NOT mention any livelihood system, rainfall season, or shock 
that is not explicitly provided in the DOMAIN KNOWLEDGE CONTEXT section above. 
If information is missing, state "Insufficient data" rather than inferring.
```

---

## **3. Shock detection must occur BEFORE prompting the LLM**

Right now shock detection happens implicitly through prompting.

### **Required fix: structured shock detection pipeline**

Create a new method in `FEWSSystem`:

```python
def _detect_validated_shocks(self, context: str, region: str, assessment: RegionRiskAssessment) -> List[str]:
    """
    Detect shocks using structured keyword matching.
    Returns ONLY shocks found in shock_ontology.json.
    """
    # 1. Run keyword detection on context
    detected = self.domain_knowledge.detect_shocks(context)
    
    # 2. Filter by confidence threshold
    validated_shocks = [
        shock_type for shock_type, confidence in detected 
        if confidence >= 0.3  # Minimum confidence threshold
    ]
    
    # 3. Map to driver names
    shock_to_driver = {
        "drought": "Drought/Rainfall deficit",
        "conflict": "Conflict and insecurity",
        # ... etc
    }
    
    drivers = [shock_to_driver.get(s, s) for s in validated_shocks if s in shock_to_driver]
    
    return drivers
```

Then in `function2_explain_why`:
- Call `_detect_validated_shocks()` BEFORE creating the prompt
- Pass the validated shock list to the prompt
- LLM cannot invent new shocks

---

## **4. Interventions sometimes too generic or missing critical domains**

Even though `driver_interventions.json` exists, it's under-utilized.

### **Required fix**

In `function3_recommend_interventions`:

1. **For each driver**, look up structured interventions from `driver_interventions.json`
2. **If driver has no mapping** → log warning: `missing_info_logger.warning(f"Driver '{driver}' has no intervention mapping")`
3. **If interventions missing** → return fallback: `"Insufficient guidance in literature for driver: {driver}"`

Ensure categories are always included when appropriate:
- life-saving food
- nutrition
- WASH
- livelihoods
- markets
- health linkages
- livestock (LEGS)
- coordination
- early recovery

Add to prompt:
```
You MUST address ALL intervention domains listed in the STRUCTURED INTERVENTION MAPPINGS 
section above. If a domain is missing from the mappings, acknowledge the gap explicitly.
```

---

## **5. System should move from "helpful guesses" → "strict IPC-aligned outputs"**

Cursor must refactor prompts so the LLM outputs follow the IPC framework strictly:

```
A. Overview
B. Livelihood System (MUST reference domain knowledge lookup)
C. Seasonal Calendar (MUST reference domain knowledge lookup)
D. Shocks (MUST reference validated shock detection list)
E. Livelihood Impacts
F. Food Access
G. Nutrition
H. IPC Alignment
I. Limitations (MUST state if domain knowledge missing)
```

No deviation. Add to prompt:
```
You MUST follow this exact structure. Do not add sections. Do not skip sections.
```

---

## **6. RAG queries need to be standardized**

Current vector retrieval logic:
- inconsistent `k` values (sometimes 5, sometimes 10)
- inconsistent filters
- inconsistent query construction
- duplicates chunks
- retrieves entire documents sometimes

### **Required fix**

Create a unified retrieval function in `FEWSSystem`:

```python
def _retrieve_context(
    self, 
    query: str, 
    vectorstore: Chroma, 
    k: int = 6,
    min_chunks: int = 3
) -> Tuple[List[Document], str]:
    """
    Unified context retrieval with deduplication and validation.
    
    Returns:
        (documents, context_string)
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(query)
    
    # Check minimum chunks
    if len(docs) < min_chunks:
        return [], ""
    
    # Deduplicate by content hash
    seen = set()
    unique_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:100])  # First 100 chars as hash
        if content_hash not in seen:
            seen.add(content_hash)
            unique_docs.append(doc)
    
    # Build context (normalized chunk size)
    context = "\n\n".join([
        f"[Source: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:600]}"
        for doc in unique_docs
    ])
    
    return unique_docs, context
```

Use this function in both Function 2 and Function 3.

---

## **7. Fuzzy region matching edge cases**

Some regions fail because:
- inconsistent casing
- spelling differences
- multiple zones share names

### **Required fix**

Install `rapidfuzz` and update `ipc_parser.py`:

```python
from rapidfuzz import fuzz

def get_region_assessment(self, region: str) -> Optional[RegionRiskAssessment]:
    """Get assessment with fuzzy matching."""
    region_lower = region.lower().strip()
    
    assessments = self.identify_at_risk_regions()
    
    # 1. Exact match
    for assessment in assessments:
        if assessment.region.lower() == region_lower:
            return assessment
    
    # 2. Fuzzy match with threshold
    best_match = None
    best_score = 0
    
    for assessment in assessments:
        score = fuzz.WRatio(region_lower, assessment.region.lower())
        if score > best_score and score >= 80:  # 80% similarity threshold
            best_score = score
            best_match = assessment
    
    return best_match
```

---

## **8. Improve code structure & reliability**

You must refactor code for:
- type hints everywhere
- consistent exception classes
- no silent errors
- functions < 75 lines
- central config file for constants
- unified logging

### **Required fixes**

1. **Create `src/config.py`:**
```python
# Configuration constants
MIN_RELEVANT_CHUNKS = 3
MAX_CONTEXT_CHUNKS = 6
CHUNK_SIZE = 600
SHOCK_CONFIDENCE_THRESHOLD = 0.3
FUZZY_MATCH_THRESHOLD = 80
```

2. **Create `src/exceptions.py`:**
```python
class FEWSException(Exception):
    """Base exception for FEWS system."""
    pass

class RegionNotFoundError(FEWSException):
    """Region not found in IPC data."""
    pass

class InsufficientDataError(FEWSException):
    """Insufficient data for analysis."""
    pass

class VectorStoreError(FEWSException):
    """Vector store error."""
    pass
```

3. **Add type hints to all functions:**
```python
def function2_explain_why(
    self, 
    region: str,
    assessment: Optional[RegionRiskAssessment] = None
) -> Dict[str, Any]:
    """..."""
```

4. **Break down large functions:**
   - If function > 75 lines, split into smaller helper functions
   - Each function should do ONE thing

5. **Replace silent errors with exceptions:**
```python
# Instead of:
if not docs:
    return {"error": "..."}

# Use:
if not docs:
    raise InsufficientDataError(f"Insufficient documents retrieved for {region}")
```

---

# 🧠 **WHAT THIS PROJECT IS (manual for Cursor)**

FEWS is a Famine Early Warning System for Ethiopia that:

1. **Identifies at-risk regions** from IPC classification CSV data
2. **Explains why** regions are at risk using RAG on situation reports
3. **Recommends interventions** using RAG on intervention literature

**Key Components:**
- `src/ipc_parser.py` - Parses IPC CSV, identifies at-risk regions
- `src/document_processor.py` - Processes PDFs into chunks
- `src/domain_knowledge.py` - Structured lookups (livelihoods, seasons, shocks, interventions)
- `src/fews_system.py` - Core orchestration with 3 functions
- `fews_cli.py` - CLI interface

**Data Sources:**
- `ipc classification data/ipcFic_data.csv` - IPC phases
- `current situation report/*.pdf` - Situation reports
- `intervention-literature/*.pdf` - Intervention guidance
- `data/domain/*` - Domain knowledge (CSV/JSON)

**Vector Stores:**
- `chroma_db_reports/` - Situation report embeddings
- `chroma_db_interventions/` - Intervention literature embeddings

---

# 🚀 **YOUR TASKS AS CURSOR**

Refactor the entire FEWS project to satisfy the following:

---

## **A. No hallucinations**

LLM must be grounded ONLY in:
- retrieved text (from vector stores)
- validated domain knowledge (from CSV/JSON files)
- IPC data (from CSV)

If info missing → respond with structured "insufficient data" message.

**Implementation:**
- Add retrieval sufficiency checks
- Enforce domain knowledge lookups
- Add explicit "insufficient data" handling in prompts

---

## **B. Hard domain knowledge enforcement**

- Hard code livelihood + rainfall season lookups
- Hard code shock ontology detection
- NEVER allow LLM to "infer" a livelihood or rainfall season

**Implementation:**
- In `function2_explain_why`, get livelihood/season BEFORE prompting
- Pass as explicit context to prompt
- Add prompt constraint: "Do not infer livelihoods or seasons"

---

## **C. Rewrite prompts**

Rewrite prompts in `fews_system.py` such that:
- They are **strict, constrained**, and hierarchical
- They reference domain knowledge explicitly
- They forbid referencing "unstated assumptions"

**Implementation:**
- Add explicit "DOMAIN KNOWLEDGE CONTEXT" section to prompts
- Add "CRITICAL CONSTRAINTS" section forbidding inference
- Make output format mandatory (no deviations)

---

## **D. Improve RAG completeness + context management**

- Remove duplicate chunks
- Normalize whitespace
- Trim to max tokens

**Implementation:**
- Create `_retrieve_context()` unified function
- Add deduplication logic
- Standardize chunk size (600 chars)

---

## **E. Strengthen intervention logic**

- Structured driver → intervention mapping
- Multi-domain intervention coverage
- Differentiate immediate vs. medium-term responses
- IPC-phase alignment logic strengthened

**Implementation:**
- Look up interventions for EVERY driver
- Log missing mappings
- Ensure all 8 domains covered
- Add IPC phase prioritization logic

---

## **F. Clean codebase**

- Rewrite IPC parser for clarity
- Move constants to `config.py`
- Add proper exceptions (e.g., `RegionNotFoundError`)
- Add unit tests for each subsystem
- Improve CLI feedback

**Implementation:**
- Create `src/config.py` with all constants
- Create `src/exceptions.py` with custom exceptions
- Add type hints everywhere
- Split large functions
- Add error handling with exceptions

---

# 🎯 **OUTPUT EXPECTATION FOR CURSOR**

When Cursor begins refactoring, it should:

1. **Open each file one by one**
2. **Add missing type hints** (use `typing` module)
3. **Improve structure** (split large functions, extract helpers)
4. **Fix pipelines** (unified retrieval, structured shock detection)
5. **Rewrite prompts** (add constraints, enforce domain knowledge)
6. **Add retrieval checks** (minimum chunks, sufficiency validation)
7. **Integrate domain knowledge properly** (lookups before prompting)
8. **Add logging & exceptions** (no silent errors)
9. **Strengthen data flow** (validate inputs, check outputs)
10. **Test everything** (verify no functionality broken)

You MUST NOT change the intended outputs, only make them *more correct, grounded, and reliable*.

---

# 📌 **FINAL NOTE FOR CURSOR**

Your priority is **accuracy, safety, domain-correctness, IPC framework alignment, and zero hallucinations**.
Not stylistic changes.

**Key principles:**
- If domain knowledge says "X", use X. Don't infer Y.
- If retrieval returns < 3 chunks, say "insufficient data". Don't guess.
- If shock not detected via keywords, don't mention it. Don't infer.
- If intervention mapping missing, log it. Don't invent.

**Success criteria:**
- Zero hallucinations in test runs
- All domain knowledge properly enforced
- All retrieval checks in place
- All prompts constrained and explicit
- Code is clean, typed, and maintainable

---

# ✅ **REFACTOR CHECKLIST**

- [ ] Create `src/config.py` with constants
- [ ] Create `src/exceptions.py` with custom exceptions
- [ ] Add type hints to all functions
- [ ] Create `_retrieve_context()` unified retrieval function
- [ ] Create `_detect_validated_shocks()` structured shock detection
- [ ] Add retrieval sufficiency checks (MIN_RELEVANT_CHUNKS)
- [ ] Rewrite Function 2 prompt with hard domain knowledge constraints
- [ ] Rewrite Function 3 prompt with structured intervention mappings
- [ ] Enforce livelihood/season lookups BEFORE prompting
- [ ] Add fuzzy matching with rapidfuzz (threshold 80)
- [ ] Split large functions (>75 lines) into smaller helpers
- [ ] Replace silent errors with exceptions
- [ ] Add logging for missing data/mappings
- [ ] Standardize chunk sizes and retrieval parameters
- [ ] Add deduplication logic for retrieved chunks
- [ ] Test all three functions work correctly

---

**You can now paste this into Cursor and get a perfect refactor.**

