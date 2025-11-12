# FEWS Agent: Food Security Early Warning System

A unified system for identifying at-risk regions and recommending intervention steps for food security in Ethiopia.

## Core Functions

1. **Identify regions at risk** - Predicts which regions are at risk of food insecurity
2. **Recommend intervention steps** - Provides evidence-based intervention recommendations

## Quick Start

### 1. Install Dependencies

```bash
cd /Users/shreyaskamath/FEWS
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Start Ollama

```bash
ollama serve  # Keep this running
```

In another terminal:
```bash
ollama pull llama3.2
```

### 3. Organize Your Data

**Documents (PDFs)**: Already in `documents/` folder ✅

**Price Data**: Already in `data/raw/prices/ethiopia/wfp_prices.csv` ✅

**IPC Phase Data** (REQUIRED): Upload to `data/raw/ipc/ethiopia/ipc_phases.csv`
- Format: `date,admin2,ipc_phase,population_affected`
- Example:
  ```csv
  date,admin2,ipc_phase,population_affected
  2024-01-01,Tigray,4,1200000
  2024-01-01,Amhara,3,850000
  ```

**Optional Data** (improves accuracy):
- `data/raw/climate/ethiopia/rainfall.csv` - Rainfall data
- `data/raw/acled/ethiopia/conflict.csv` - Conflict incidents
- `data/raw/population/ethiopia/population.csv` - Population data

See `DATA_REQUIREMENTS.md` for detailed format requirements.

### 4. Run the System

```bash
python3 main.py
```

### 5. Use the System

**Identify at-risk regions:**
```
Query: risk
```

**Get intervention steps:**
```
Query: interventions Tigray
```

**General queries (still work):**
```
Query: What are current maize prices in Addis Ababa?
```

## Web Interface

```bash
python3 app.py
```

Then open: http://localhost:5001

## Project Structure

```
FEWS/
├── main.py                    # Unified entry point
├── rag_system.py              # Unified RAG (risk + interventions)
├── data_loader.py             # Load CSV files manually
├── document_processor.py      # Process PDFs
├── models/
│   ├── risk_predictor.py      # Risk prediction
│   ├── feature_engineering.py # Feature extraction
│   └── intervention_recommender.py # Intervention mapping
├── data/
│   └── raw/
│       ├── prices/ethiopia/wfp_prices.csv      ✅ You have
│       ├── ipc/ethiopia/ipc_phases.csv         ⬆️ Upload
│       ├── climate/ethiopia/rainfall.csv        ⬆️ Upload (optional)
│       └── acled/ethiopia/conflict.csv         ⬆️ Upload (optional)
├── documents/                 # PDFs here
│   └── *.pdf
├── kb/
│   └── interventions/
│       └── playbook.yaml      # Intervention rules
└── chroma_db/                 # Vector store (existing)
```

## Data Requirements

See `DATA_REQUIREMENTS.md` for complete details on:
- What data you need
- What you already have
- CSV file formats
- Where to get data

## Troubleshooting

**"No data available for risk assessment"**
- Upload IPC phase data to `data/raw/ipc/ethiopia/ipc_phases.csv`

**"Vector store not found"**
- Documents will be processed automatically on first run

**"Ollama not running"**
- Run `ollama serve` in a separate terminal

## Notes

- System works with just price data + IPC phases (minimum)
- Additional data (rainfall, conflict) improves accuracy
- Documents provide context for better recommendations
- All data stays local (privacy-preserving)
