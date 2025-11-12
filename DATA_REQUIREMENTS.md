# Data Requirements for FEWS Agent

## What You Have ✅

1. **Price Data**: `data/raw/prices/ethiopia/wfp_prices.csv`
   - Status: ✅ Ready to use
   - Contains: date, admin1, admin2, market, commodity, price, usdprice

2. **Documents**: `documents/*.pdf`
   - Status: ✅ Processed (15,440 chunks in vector store)
   - Contains: IPC reports, FEWS bulletins

## What You Need to Upload

### REQUIRED (for risk identification)

#### 1. IPC Phase Data
**File**: `data/raw/ipc/ethiopia/ipc_phases.csv`

**Required Columns**:
- `date` - Date in YYYY-MM-DD format
- `admin2` - Region name (e.g., "Tigray", "Amhara")
- `ipc_phase` - IPC phase (1-5)
- `population_affected` - Number of people affected (optional but recommended)

**Example**:
```csv
date,admin2,ipc_phase,population_affected
2024-01-01,Tigray,4,1200000
2024-01-01,Amhara,3,850000
2024-02-01,Tigray,4,1250000
2024-02-01,Amhara,3,900000
```

**Where to get**: Download from HDX (https://data.humdata.org/organization/ipc) or IPC website

---

### OPTIONAL (improves accuracy)

#### 2. Rainfall Data
**File**: `data/raw/climate/ethiopia/rainfall.csv`

**Columns**:
- `date` - Date in YYYY-MM-DD format
- `admin2` - Region name
- `rainfall_mm` - Rainfall in millimeters

**Example**:
```csv
date,admin2,rainfall_mm
2024-01-01,Tigray,45.2
2024-01-01,Amhara,52.1
```

#### 3. Conflict Data
**File**: `data/raw/acled/ethiopia/conflict.csv`

**Columns**:
- `date` - Date in YYYY-MM-DD format
- `admin2` - Region name
- `conflict_incidents` - Number of conflict events (or will be counted)

**Example**:
```csv
date,admin2,conflict_incidents
2024-01-15,Tigray,5
2024-01-20,Amhara,2
```

#### 4. Population Data
**File**: `data/raw/population/ethiopia/population.csv`

**Columns**:
- `admin2` - Region name
- `population` - Total population

**Example**:
```csv
admin2,population
Tigray,5200000
Amhara,21000000
```

---

## Intervention Playbook

**File**: `kb/interventions/playbook.yaml`

Create this file with intervention rules. Example:

```yaml
interventions:
  - driver: "price_increase"
    intervention: "Market support and cash transfers"
    ipc_phase_applicable: [3, 4, 5]
    ops_requirements: ["market access", "cash transfer system"]
    evidence_link: "IPC Technical Manual"
  
  - driver: "rainfall_deficit"
    intervention: "Emergency food assistance"
    ipc_phase_applicable: [4, 5]
    ops_requirements: ["food stocks", "distribution network"]
  
  - driver: "conflict_escalation"
    intervention: "Protection services and emergency food"
    ipc_phase_applicable: [4, 5]
    ops_requirements: ["security", "mobile distribution"]
  
  - driver: "multi_driver"
    intervention: "Multi-sectoral response"
    ipc_phase_applicable: [4, 5]
    ops_requirements: ["coordination", "multiple sectors"]
```

---

## Quick Start

1. **Minimum setup** (risk identification works):
   - Upload `data/raw/ipc/ethiopia/ipc_phases.csv`
   - Price data already exists ✅

2. **Full setup** (better accuracy):
   - Upload all optional data files
   - Create intervention playbook

3. **Run system**:
   ```bash
   python3 main_unified.py
   ```

---

## Data Format Notes

- **Dates**: Use YYYY-MM-DD format
- **Region names**: Must match between files (case-insensitive matching)
- **Missing data**: Use empty cells or NaN (system handles gracefully)
- **Encoding**: UTF-8 CSV files

