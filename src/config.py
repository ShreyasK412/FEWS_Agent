"""
Configuration constants for FEWS system.
"""
# Retrieval configuration
MIN_RELEVANT_CHUNKS = 3
MAX_CONTEXT_CHUNKS = 6
CHUNK_SIZE = 600
CHUNK_OVERLAP = 200

# Shock detection configuration
SHOCK_CONFIDENCE_THRESHOLD = 0.3
MAX_SHOCKS_TO_RETURN = 8

# Fuzzy matching configuration
FUZZY_MATCH_THRESHOLD = 80  # Percentage similarity (0-100)
MAX_FUZZY_CANDIDATES = 3

# Domain knowledge paths
DOMAIN_DATA_DIR = "data/domain"
LIVELIHOOD_SYSTEMS_FILE = "data/domain/livelihood_systems.csv"
RAINFALL_SEASONS_FILE = "data/domain/rainfall_seasons.csv"
SHOCK_ONTOLOGY_FILE = "data/domain/shock_ontology.json"
DRIVER_INTERVENTIONS_FILE = "data/domain/driver_interventions.json"

# IPC data paths
IPC_DATA_FILE = "ipc classification data/ipcFic_data.csv"
REPORTS_DIR = "current situation report"
INTERVENTIONS_DIR = "intervention-literature"

# Vector store paths
VECTOR_STORE_REPORTS = "chroma_db_reports"
VECTOR_STORE_INTERVENTIONS = "chroma_db_interventions"

# LLM configuration
DEFAULT_MODEL_NAME = "llama3.2"

# Logging
MISSING_INFO_LOG_FILE = "missing_info.log"

