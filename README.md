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

# 4. (Optional) Launch the Streamlit review + agent UI
streamlit run review_app.py
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

## Web UI (Streamlit) – Human Review + FEWS Agent

In addition to the CLI, the project exposes a **Streamlit UI** that lets you:

- **Review and correct low-confidence outputs** (human-in-the-loop).
- **Approve human-reviewed explanations** that become ground truth for future runs.
- **Interact with the FEWS agent** via a point-and-click interface that mirrors the CLI actions (identify at-risk regions, explain why, recommend interventions, or run a full analysis).

### Launching the UI

```bash
# From the FEWS project root, in your virtualenv
pip install -r requirements.txt
streamlit run review_app.py
```

Then open the URL shown in the terminal (typically `http://localhost:8501`).

### Tabs

- **Human review**
  - Reads `human_review_queue.jsonl` for **pending low-confidence explanations** automatically enqueued by `function2_explain_why` (e.g., insufficient data or only low-confidence shocks).
  - Lets an analyst:
    - Inspect model-detected shocks (type, description, evidence, confidence),
    - Inspect drivers and sources,
    - Edit shocks as JSON, edit drivers (comma-separated), and edit the explanation text,
    - Save a **human-approved record** to `human_review_approved.jsonl`.
  - On subsequent runs, `FEWSSystem.function2_explain_why` checks `human_review_approved.jsonl` via `find_approved_explanation` and, if a match exists for `(region, IPC phase)`, **returns the human-reviewed explanation/drivers/shocks instead of re-running RAG + LLM**, with `data_quality="human_validated"`.

- **Chat with FEWS agent**
  - A **CLI-style agent UI** that mirrors the CLI menu:
    - `1 - Identify at-risk regions`
    - `2 - Explain why a region is at risk`
    - `3 - Recommend interventions for a region`
    - `4 - Full analysis (risk + why + interventions)`
  - Workflow:
    1. Click **Initialize FEWS system** to construct `FEWSSystem()` (loads IPC parser, domain knowledge, and LLM/embeddings).
    2. Click **Setup vector stores** to build/load the Chroma stores for situation reports and intervention literature.
    3. Choose an action (1–4).
    4. For actions 2–4, pick a **region from a dropdown** populated from IPC data (`IPCParser.identify_at_risk_regions(min_phase=1, ...)`).
    5. Click **Run action** to execute the corresponding FEWS functions and display:
       - At-risk region tables (for action 1),
       - Full explanations with IPC phase, data_quality, drivers, and sources (action 2),
       - Intervention recommendations (action 3),
       - Combined risk assessment + explanation + interventions (action 4).

The web UI and CLI share the same core logic in `src/fews_system.py`, `src/ipc_parser.py`, and `src/domain_knowledge.py`, so behavior is consistent regardless of interface.

---

## Human-in-the-Loop Review Design

Low-confidence explanations from `function2_explain_why` are **automatically queued** for human review:

- The logic in `src/human_review.py` uses:
  - `should_enqueue_for_review(data_quality, shocks)` to decide when to enqueue.
  - `enqueue_explanation_for_review(...)` to append items to `human_review_queue.jsonl`.

Each queue item contains:

- `region`, `ipc_phase`, `geographic_full_name`
- `data_quality`
- `shocks_model` (list of shock dicts with type/description/evidence/confidence)
- `drivers_model`
- `explanation_model` (LLM text)
- `sources_model`
- `status="pending"`, `created_at`.

When an analyst approves an item in the Streamlit UI:

- A **human-reviewed record** is written to `human_review_approved.jsonl` with:
  - `shocks_human`, `drivers_human`, `explanation_human`, `sources_human`
  - `reviewed_by`, `reviewed_at`.

On subsequent runs:

- `FEWSSystem.function2_explain_why` calls `find_approved_explanation(region, ipc_phase)`:
  - If found, it **skips retrieval/shock detection/LLM and returns the human-approved explanation**, shocks, and drivers as the authoritative ground truth.
  - Otherwise, it runs the normal RAG + domain-knowledge pipeline and may enqueue a new item if confidence is low.

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
