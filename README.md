# FEWS: Famine Early Warning System

A RAG-based system for identifying at-risk regions and recommending evidence-based interventions for food security in Ethiopia. Uses structured domain knowledge and LLM inference for accurate, consistent analysis.

## Overview

FEWS provides three core functions:

1. **Identify At-Risk Regions** - Analyzes IPC classification data to flag regions with IPC Phase 3+ or deteriorating trends
2. **Explain Why** - Uses RAG on situation reports with structured domain knowledge to explain the drivers of food insecurity (conflict, drought, prices, displacement, etc.)
3. **Recommend Interventions** - Uses RAG on intervention literature with driver-intervention mappings to provide practical, evidence-based recommendations

## Key Features

- **Structured Domain Knowledge**: Livelihood systems, rainfall seasons, shock ontology, and intervention mappings ensure accurate analysis
- **Fuzzy Region Matching**: Handles typos, partial names, and case variations with intelligent suggestions
- **IPC-Aligned Analysis**: Follows IPC analytical frameworks for consistent, professional outputs
- **Evidence-Based**: All recommendations grounded in actual documents with citations
- **Missing Data Logging**: Tracks queries with insufficient information for data quality monitoring

## System Architecture

- **IPC Parser** (`src/ipc_parser.py`) - Parses IPC classification CSV and identifies at-risk regions with fuzzy matching
- **Document Processor** (`src/document_processor.py`) - Processes PDFs into chunks for vector storage
- **Domain Knowledge** (`src/domain_knowledge.py`) - Provides structured lookups for livelihoods, seasons, shocks, and interventions
- **FEWS System** (`src/fews_system.py`) - Core orchestration with RAG pipelines and domain knowledge integration
- **CLI Interface** (`fews_cli.py`) - Interactive command-line interface

## Requirements

- Python 3.9+
- Ollama (local LLM server)
- Llama3.2 model (or compatible)

## Installation

### 1. Install Python Dependencies

```bash
cd /Users/shreyaskamath/FEWS
source venv/bin/activate  # or create new venv: python3 -m venv venv
pip install -r requirements.txt
```

### 2. Install and Start Ollama

```bash
# Install Ollama from https://ollama.ai/download

# Start Ollama server (keep running)
ollama serve

# In another terminal, pull the model
ollama pull llama3.2
```

## Data Setup

The system uses **three data sources**:

### 1. IPC Classification Data (REQUIRED)

**Location:** `ipc classification data/ipcFic_data.csv`

**Format:** CSV with columns:
- `admin2` - Region name (e.g., "Burji Special")
- `date` - Date of classification
- `scenario` - "CS" (current situation), "ML1" (near-term projection), or "ML2" (medium-term projection)
- `value` - IPC Phase (1-5)
- `projection_end` - End date for projections

**Example:**
```csv
admin2,date,scenario,value,projection_end
Burji Special,2025-01-01,CS,4,
Burji Special,2025-02-01,ML1,4,2025-04-30
```

### 2. Situation Reports (REQUIRED for "Explain Why")

**Location:** `current situation report/*.pdf`

**Content:** PDF reports explaining the context, drivers, and conditions causing food insecurity. These are processed into a vector database for RAG queries.

**Example:** `et-fso-2025-10-1762547472.pdf` (Ethiopia Food Security Outlook)

### 3. Intervention Literature (REQUIRED for "Recommend Interventions")

**Location:** `intervention-literature/*.pdf`

**Content:** PDF documents with intervention guidance, best practices, and protocols (e.g., Sphere Handbook, WFP guidelines, CMAM protocols, LEGS standards).

**Example:** `Comprehensive Report on Emergency Food Security and Nutrition Interventions.pdf`

## Domain Knowledge Layer

The system includes a structured domain knowledge layer in `data/domain/`:

- **`livelihood_systems.csv`** - Maps regions/zones to livelihood systems (rainfed cropping, pastoral, agropastoral, mixed enset)
- **`rainfall_seasons.csv`** - Maps regions to dominant rainfall seasons (Kiremt, Gu/Genna, Deyr/Hageya)
- **`shock_ontology.json`** - Defines shock types with keyword lists for structured detection
- **`driver_interventions.json`** - Maps drivers to structured intervention categories

This ensures:
- ✅ Correct livelihood identification (no "pastoral Tigray" errors)
- ✅ Correct rainfall season usage (Kiremt for highlands, deyr/hageya for pastoral)
- ✅ Structured shock detection (keyword-based, not hallucinated)
- ✅ Complete intervention coverage (all 8 domains mapped)

## Usage

### Command-Line Interface

```bash
python3 fews_cli.py
```

**Menu Options:**

1. **Identify at-risk regions** - Lists all regions with IPC 3+ or deteriorating trends
2. **Explain why a region is at risk** - Provides detailed analysis of drivers (conflict, drought, prices, etc.)
3. **Recommend interventions for a region** - Generates evidence-based intervention recommendations
4. **Full analysis** - Runs all three functions for a complete assessment
5. **Exit**

### Region Name Matching

The system supports fuzzy matching for region names:
- **Exact match** (case-insensitive): "Burji Special" → matches "Burji Special"
- **Partial match**: "burji" → matches "Burji Special"
- **Fuzzy match**: "Edaga arb" → matches "Edaga arbi" (similarity-based)

If a region isn't found, the CLI suggests similar region names.

### Example Workflow

```bash
# Start the CLI
python3 fews_cli.py

# Select option 1 to see at-risk regions
# Select option 4 for full analysis of "Edaga arbi" (or "Edaga arb" - fuzzy matching works!)
```

### Programmatic Usage

```python
from src import FEWSSystem

# Initialize system
system = FEWSSystem()

# Setup vector stores (first time: 30-60 minutes for large PDFs)
system.setup_vector_stores()

# Function 1: Identify at-risk regions
at_risk = system.function1_identify_at_risk_regions()
print(f"Found {len(at_risk)} at-risk regions")

# Function 2: Explain why (with domain knowledge)
assessment = system.ipc_parser.get_region_assessment("Edaga arbi")  # Fuzzy matching works!
explanation = system.function2_explain_why("Edaga arbi", assessment)
print(explanation['explanation'])
print(f"Drivers: {explanation['drivers']}")

# Function 3: Recommend interventions (with structured mappings)
interventions = system.function3_recommend_interventions(
    "Edaga arbi", 
    assessment, 
    explanation.get('drivers', [])
)
print(interventions['recommendations'])
```

## Project Structure

```
FEWS/
├── fews_cli.py                  # CLI interface (main entry point)
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── src/                         # Source code package
│   ├── __init__.py             # Package initialization
│   ├── fews_system.py          # Core system with 3 functions
│   ├── ipc_parser.py           # IPC data parsing and risk identification
│   ├── document_processor.py   # PDF processing and chunking
│   └── domain_knowledge.py     # Domain knowledge lookup class
│
├── tests/                       # Test suite
│   ├── __init__.py
│   └── test_fews.py            # Basic test script
│
├── data/                        # Domain knowledge layer
│   └── domain/
│       ├── livelihood_systems.csv      # Livelihood system mappings
│       ├── rainfall_seasons.csv        # Rainfall season mappings
│       ├── shock_ontology.json         # Shock type definitions
│       └── driver_interventions.json   # Driver-intervention mappings
│
├── ipc classification data/     # IPC CSV data (REQUIRED)
│   └── ipcFic_data.csv
│
├── current situation report/    # Situation report PDFs (REQUIRED)
│   └── *.pdf
│
├── intervention-literature/     # Intervention PDFs (REQUIRED)
│   └── *.pdf
│
├── chroma_db_reports/           # Vector store for situation reports (auto-generated)
├── chroma_db_interventions/     # Vector store for interventions (auto-generated)
└── missing_info.log             # Logs queries with insufficient data
```

## How It Works

### Function 1: Identify At-Risk Regions

- Loads IPC classification CSV
- Identifies regions with:
  - Current IPC Phase ≥ 3
  - Projected IPC Phase ≥ 3 (ML1 or ML2)
  - Deteriorating trends (projected phase > current phase)
- Returns structured risk assessments with geographic hierarchy
- Supports fuzzy region name matching

### Function 2: Explain Why

**Domain Knowledge Integration:**
- Looks up livelihood system from `livelihood_systems.csv`
- Looks up rainfall season from `rainfall_seasons.csv`
- Detects shocks using `shock_ontology.json` keyword matching

**RAG Pipeline:**
- Retrieves relevant chunks from situation report vector store
- Uses expanded region variations (zone, region, country) and livelihood keywords
- LLM analyzes using IPC-aligned framework:
  - Livelihood system identification (from domain knowledge)
  - Rainfall season context (from domain knowledge)
  - Shock identification (from domain knowledge + context)
  - Livelihood impacts
  - Food access gaps
  - Nutrition outcomes
  - IPC phase alignment
- Logs "insufficient data" cases to `missing_info.log`

### Function 3: Recommend Interventions

**Domain Knowledge Integration:**
- Maps drivers to structured interventions using `driver_interventions.json`
- Ensures coverage of all 8 intervention domains

**RAG Pipeline:**
- Retrieves relevant chunks from intervention literature vector store
- Uses generic intervention keywords + driver-specific keywords
- LLM generates recommendations using:
  - Driver-linked intervention mapping (from domain knowledge)
  - IPC phase-specific prioritization
  - Best practice references (Sphere, WFP, CMAM, LEGS)
  - Structured output (Immediate Actions, Food & Nutrition, Cash Support, etc.)
- Provides citations from intervention literature

## Vector Store Setup

**First Run:** Vector stores are created automatically when you call `system.setup_vector_stores()`. This can take:
- Small PDFs (< 50 pages): 5-10 minutes
- Medium PDFs (50-200 pages): 15-30 minutes
- Large PDFs (> 200 pages): 30-60+ minutes

**Subsequent Runs:** Vector stores are loaded instantly if they already exist.

**To Regenerate:** Delete the vector store directories and restart:
```bash
rm -rf chroma_db_reports/ chroma_db_interventions/
python3 fews_cli.py
```

## Performance Optimization

The system is optimized for accuracy and consistency. For faster performance:

1. **Reduce retrieval chunks**: Change `k=10` to `k=6` in `fews_system.py` (lines 407, 774)
2. **Reduce context size**: Change `[:800]` to `[:600]` in `fews_system.py` (lines 432, 781)
3. **Use faster model**: Try `llama3.2:1b` or `phi3` for faster inference (may reduce quality)

## Troubleshooting

### "IPC file not found"
- Ensure `ipc classification data/ipcFic_data.csv` exists
- Check the file path matches the default: `ipc classification data/ipcFic_data.csv`

### "Region not found"
- Try partial names: "burji" instead of "Burji Special"
- Try fuzzy matching: "Edaga arb" matches "Edaga arbi"
- Check suggestions provided by the CLI

### "No PDFs found in reports/interventions directories"
- Ensure PDFs are in `current situation report/` and `intervention-literature/`
- Check file permissions

### "Ollama not running"
- Start Ollama: `ollama serve`
- Verify model is pulled: `ollama list`
- Pull model if needed: `ollama pull llama3.2`

### "Vector store is empty"
- Delete and regenerate: `rm -rf chroma_db_reports/ chroma_db_interventions/`
- Restart the system to recreate vector stores

### "Insufficient data" warnings
- Check `missing_info.log` for logged queries
- Add more situation reports or intervention literature
- Verify PDFs contain relevant content

### "Domain knowledge files not found"
- Ensure `data/domain/` directory exists with all 4 files
- System will use fallback logic if files are missing (less accurate)

### Long processing time
- Normal for first-time vector store creation
- Large PDFs (> 200 pages) take longer
- Progress indicators show estimated time
- See Performance Optimization section above

## Data Quality

The system explicitly handles missing information:

- **"Explain Why"** - If no relevant information found, states "insufficient data" and logs to `missing_info.log`
- **"Recommend Interventions"** - If guidance is insufficient, states limitations clearly
- **Never hallucinates** - All responses are grounded in actual documents with citations
- **Domain knowledge ensures accuracy** - Structured lookups prevent common errors (wrong livelihoods, wrong seasons)

## Testing

```bash
python3 tests/test_fews.py
```

This runs basic IPC parsing tests. For full testing, use the CLI interface.

## Notes

- All data stays local (privacy-preserving)
- System works offline once vector stores are created
- IPC data must be updated regularly for accurate risk identification
- Situation reports and intervention literature should be current and relevant
- The system uses case-insensitive region matching with fuzzy fallback
- Domain knowledge files can be updated to improve accuracy

## Recent Updates

- **Domain Knowledge Layer**: Added structured lookups for livelihoods, seasons, shocks, and interventions
- **Fuzzy Region Matching**: Handles typos and partial names with intelligent suggestions
- **Improved Prompts**: Added livelihood mapping rules and rainfall season rules to prevent errors
- **Bug Fixes**: Fixed variable reference errors and improved error handling

## License

[Your License Here]

## Contact

[Your Contact Information]
