# FEWS – Famine Early Warning System (Ethiopia)

A local Retrieval-Augmented Generation (RAG) CLI that analyzes Ethiopian IPC data, FEWS NET situation reports, and humanitarian intervention literature to:

1. **Identify regions at risk** (IPC Phase ≥ 3 or deteriorating trends)
2. **Explain why** those areas are food insecure (validated shocks, livelihoods, seasons)
3. **Recommend interventions** grounded in Sphere / WFP / FAO / LEGS / CALP guidance

The system combines structured domain knowledge with LLM reasoning, prioritising factual consistency over hallucination.

---

## Quick Start

```bash
# 1. Install dependencies (Python 3.9+)
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install + start Ollama (https://ollama.ai)
ollama serve        # keep running in a separate terminal
ollama pull llama3.2

# 3. Launch the CLI
python3 fews_cli.py
```

When prompted, choose an option (1–4) and enter a woreda name, e.g. `Edaga arbi`. Fuzzy matching handles typos like `Edaga arb`.

---

## Data Inputs

| Type | Location | Required For | Notes |
|------|----------|--------------|-------|
| IPC classification CSV | `ipc classification data/ipcFic_data.csv` | Function 1 (+2/3 for context) | Include `admin2`, `date`, `scenario` (CS/ML1/ML2), `value` (IPC 1–5) |
| Situation reports (PDF) | `current situation report/` | Function 2 | FEWS NET Ethiopia Food Security Outlook (e.g. `et-fso-2025-10-1762547472.pdf`) |
| Intervention literature (PDF) | `intervention-literature/` | Function 3 | e.g. Sphere Handbook extracts, WFP/CALP guidance |
| Domain knowledge CSV/JSON | `data/domain/` | All functions | `livelihood_systems.csv`, `rainfall_seasons.csv`, `shock_ontology.json`, `driver_interventions.json` |

Vector stores for the PDFs are generated automatically on first run (can take 15–30 minutes for large reports). Delete `chroma_db_*` directories to rebuild.

---

## CLI Overview

```
1. Identify at-risk regions
2. Explain why a region is at risk
3. Recommend interventions for a region
4. Full analysis (1 + 2 + 3)
5. Exit
```

- **Function 1**: returns IPC-driven risk assessments (current + projections) with fuzzy admin2 matching.
- **Function 2**: blends retrieval with structured domain knowledge (livelihood, rainfall, shocks) to generate IPC-aligned explanations, explicitly reporting limitations when evidence is thin.
- **Function 3**: maps validated drivers (economic, conflict, weather, displacement) into intervention packages from `driver_interventions.json`, referencing humanitarian standards.

All outputs are logged; retrieval warnings and “insufficient data” cases are recorded for follow-up.

---

## Repository Layout

```
FEWS/
├── fews_cli.py                # CLI entry point
├── fews_system.py             # Convenience re-export (imports src.FewsSystem)
├── requirements.txt
├── README.md                  # This document
│
├── src/
│   ├── __init__.py            # Exposes FEWSSystem, RegionRiskAssessment
│   ├── fews_system.py         # Core orchestration (RAG + domain knowledge)
│   ├── ipc_parser.py          # IPC CSV ingestion + fuzzy matching
│   ├── document_processor.py  # PDF chunking for vector stores
│   └── domain_knowledge.py    # Loads CSV/JSON knowledge tables
│
├── data/domain/               # Static knowledge tables
│   ├── livelihood_systems.csv
│   ├── rainfall_seasons.csv
│   ├── shock_ontology.json
│   └── driver_interventions.json
│
├── ipc classification data/   # IPC CSVs (user-provided)
├── current situation report/  # FEWS NET PDFs (user-provided)
├── intervention-literature/   # Intervention PDFs (user-provided)
├── tests/                     # Basic IPC parser smoke test
├── test_shock_detection.py    # Manual diagnostic script
└── check_vector_store.py      # Vector store health check
```

Temporary artefacts (`__pycache__/`, `chroma_db_*`, `missing_info.log`) are generated at runtime and not tracked in Git.

---

## Tips & Troubleshooting

- **Region not found?** Try partial or fuzzy names; the CLI will list close matches.
- **No evidence retrieved?** Add more situation reports or delete/regenerate vector stores.
- **Slow first run?** Large PDFs take time; subsequent runs reuse cached vector stores.
- **Model issues?** Ensure Ollama is running, the `llama3.2` model is pulled, and your machine has enough RAM.

---

## Development

- Python ≥ 3.9
- Recommended: macOS or Linux with local LLM acceleration
- Tests: `python3 -m pytest tests`

Contributions are welcome—please open an issue describing the improvement or bug fix you have in mind.

---

## License & Contact

Copyright © 2025.

Released under the MIT License—see `LICENSE` (add if required).

Questions or support: add your preferred contact information here.
