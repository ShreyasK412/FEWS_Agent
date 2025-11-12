# Famine Early Warning System (FEWS)

A robust system for identifying at-risk regions, explaining drivers of food insecurity, and recommending interventions based on IPC classifications and situation reports.

## Core Functions

### 1. Identify At-Risk Regions
- Parses IPC classification CSV to extract current and projected phases
- Identifies regions with IPC Phase 3+ or deteriorating trends
- Returns structured risk assessments sorted by risk level

### 2. Explain Why (RAG on Situation Reports)
- Answers: "Why is region X at risk?"
- Queries vector database of situation reports
- Extracts drivers: conflict, drought, prices, displacement, etc.
- **Explicitly states when information is insufficient** (no hallucination)
- Logs all missing information to `missing_info.log`

### 3. Recommend Interventions (RAG on Intervention Literature)
- Answers: "What interventions for region X?"
- Queries vector database of intervention literature
- Generates practical recommendations for small NGOs/individuals
- Based on IPC phase and identified drivers
- States limitations when guidance is insufficient

## Data Structure

```
ipc classification data/
  └── ipcFic_data.csv          # IPC phases + projections (CS, ML1, ML2)

current situation report/
  └── *.pdf                    # Situation reports explaining context

intervention-literature/
  └── *.pdf                    # Intervention guidance documents
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure Ollama is running:**
   ```bash
   ollama serve
   ollama pull llama3.2
   ```

3. **Organize your data:**
   - Place IPC CSV in `ipc classification data/ipcFic_data.csv`
   - Place situation reports in `current situation report/`
   - Place intervention literature in `intervention-literature/`

## Usage

### Command Line Interface

```bash
python3 fews_cli.py
```

The CLI provides an interactive menu:
1. Identify at-risk regions
2. Explain why a region is at risk
3. Recommend interventions for a region
4. Full analysis (all 3 functions) for a region
5. Exit

### Programmatic Usage

```python
from fews_system import FEWSSystem

# Initialize system
system = FEWSSystem()

# Setup vector stores (first time only, may take a few minutes)
system.setup_vector_stores()

# Function 1: Identify at-risk regions
at_risk = system.function1_identify_at_risk_regions()

# Function 2: Explain why
explanation = system.function2_explain_why("Abaala", at_risk[0])

# Function 3: Recommend interventions
interventions = system.function3_recommend_interventions(
    "Abaala", 
    at_risk[0], 
    explanation.get("drivers", [])
)
```

## IPC Data Structure

The system expects IPC CSV with:
- `geographic_unit_name`: Region name
- `scenario`: CS (Current Situation), ML1 (Near-term), ML2 (Medium-term)
- `value`: IPC phase (1-5)
- `projection_start`, `projection_end`: Date ranges
- `description`: Phase description

## Risk Identification Logic

A region is flagged as at-risk if:
- Current IPC Phase >= 3, OR
- Projected Phase (ML1/ML2) >= 3, OR
- Deteriorating trend (phase increasing over time)

Risk levels:
- **HIGH**: IPC Phase 4+
- **MEDIUM**: IPC Phase 3
- **LOW**: IPC Phase 1-2

## Missing Information Logging

All cases where insufficient data is found are logged to `missing_info.log` with:
- Region name
- IPC phase
- Issue description
- Timestamp

This helps identify data gaps for future collection.

## Output Files

- `missing_info.log`: Log of all insufficient data cases
- `chroma_db_reports/`: Vector store for situation reports
- `chroma_db_interventions/`: Vector store for intervention literature

## Requirements

- Python 3.9+
- Ollama with llama3.2 model
- PDF processing: pypdf
- Vector database: chromadb
- LLM: langchain-ollama

## Notes

- **No hallucination**: System explicitly states when information is not available
- **Citation-based**: All responses cite sources from documents
- **Practical focus**: Interventions tailored for small NGOs/individuals
- **Robust parsing**: Handles IPC data structure with multiple scenarios and time periods

