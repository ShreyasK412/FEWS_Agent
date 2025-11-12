# Data Upload Guide

## What You Already Have ✅

**Price Data**: `data/raw/prices/ethiopia/wfp_prices.csv`
- ✅ Already uploaded
- Contains: date, admin1, admin2, market, commodity, price, usdprice
- Date range: 2000-01-15 onwards

---

## REQUIRED: What You Must Upload

### 1. IPC Phase Data (REQUIRED)

**File Location**: `data/raw/ipc/ethiopia/ipc_phases.csv`

**Required Columns**:
- `date` - Date in YYYY-MM-DD format (e.g., "2024-01-01")
- `admin2` - Region name (must match price data regions, e.g., "Tigray", "Amhara", "AA ZONE1")
- `ipc_phase` - IPC phase number (1, 2, 3, 4, or 5)
- `population_affected` - Number of people affected (optional but recommended)

**Example CSV Format**:
```csv
date,admin2,ipc_phase,population_affected
2024-01-01,Tigray,4,1200000
2024-01-01,Amhara,3,850000
2024-01-01,AA ZONE1,2,50000
2024-02-01,Tigray,4,1250000
2024-02-01,Amhara,3,900000
```

**Important Notes**:
- `admin2` names must match the regions in your price CSV (case-insensitive)
- Dates should overlap with your price data dates
- IPC phases: 1=Minimal, 2=Stressed, 3=Crisis, 4=Emergency, 5=Famine
- If you don't have `population_affected`, you can omit that column

**Where to Get This Data**:
- HDX: https://data.humdata.org/organization/ipc
- IPC Website: https://www.ipcinfo.org/
- Search for "Ethiopia IPC" datasets

---

## OPTIONAL: Additional Data (Improves Accuracy)

### 2. Rainfall Data (Optional)

**File Location**: `data/raw/climate/ethiopia/rainfall.csv`

**Columns**:
- `date` - Date in YYYY-MM-DD format
- `admin2` - Region name (must match other data)
- `rainfall_mm` - Rainfall in millimeters

**Example**:
```csv
date,admin2,rainfall_mm
2024-01-01,Tigray,45.2
2024-01-01,Amhara,52.1
2024-02-01,Tigray,38.5
```

**Where to Get**: CHIRPS, FEWS NET, or local meteorological data

---

### 3. Conflict Data (Optional)

**File Location**: `data/raw/acled/ethiopia/conflict.csv`

**Columns**:
- `date` - Date in YYYY-MM-DD format
- `admin2` - Region name
- `conflict_incidents` - Number of conflict events (or will be counted automatically)

**Example**:
```csv
date,admin2,conflict_incidents
2024-01-15,Tigray,5
2024-01-20,Amhara,2
2024-02-10,Tigray,8
```

**Where to Get**: ACLED (Armed Conflict Location & Event Data Project)

---

### 4. Population Data (Optional)

**File Location**: `data/raw/population/ethiopia/population.csv`

**Columns**:
- `admin2` - Region name
- `population` - Total population (static, no date needed)

**Example**:
```csv
admin2,population
Tigray,5200000
Amhara,21000000
AA ZONE1,3500000
```

**Where to Get**: WorldPop, UN Population Division, or national census data

---

## Quick Start (Minimum Setup)

**To get the system working, you only need:**

1. ✅ Price data (already have)
2. ⬆️ **IPC phase data** - Upload `data/raw/ipc/ethiopia/ipc_phases.csv`

That's it! The system will work with just these two files.

---

## Data Format Requirements

- **File Format**: CSV (comma-separated)
- **Encoding**: UTF-8
- **Date Format**: YYYY-MM-DD (e.g., "2024-01-15")
- **Region Names**: Must match between files (case-insensitive matching)
- **Missing Data**: Use empty cells or NaN (system handles gracefully)
- **Headers**: First row should be column names

---

## How to Upload

1. **Create the directory** (if needed):
   ```bash
   mkdir -p data/raw/ipc/ethiopia
   ```

2. **Save your CSV file**:
   ```bash
   # Save as: data/raw/ipc/ethiopia/ipc_phases.csv
   ```

3. **Test the upload**:
   ```bash
   python3 -c "from data_loader import DataLoader; loader = DataLoader(); loader.load_all()"
   ```

---

## What Happens If Data Is Missing?

- **No IPC data**: System cannot identify at-risk regions (will show error)
- **No price data**: System cannot work (already have this ✅)
- **No optional data**: System works but with lower accuracy

---

## Example: Complete Minimum Dataset

**File 1**: `data/raw/prices/ethiopia/wfp_prices.csv` ✅ (you have this)

**File 2**: `data/raw/ipc/ethiopia/ipc_phases.csv` ⬆️ (upload this)
```csv
date,admin2,ipc_phase,population_affected
2024-01-01,Tigray,4,1200000
2024-01-01,Amhara,3,850000
2024-01-01,AA ZONE1,2,50000
2024-02-01,Tigray,4,1250000
2024-02-01,Amhara,3,900000
2024-02-01,AA ZONE1,2,50000
```

That's all you need to get started!

