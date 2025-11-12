# FEWS Refactoring Plan - Comprehensive Implementation

## Overview
This document outlines the complete refactoring to ensure FEWS system outputs are 100% consistent with:
1. IPC dataset
2. FEWS NET Ethiopia Food Security Outlook
3. Sphere/WFP/FAO/LEGS/CALP guidance

## Changes Required

### Phase 1: Infrastructure ✅ DONE
- [x] Create `fews_knowledge_library.py` with seasonal calendars, impact templates, IPC definitions, driver mappings

### Phase 2: Function 2 Metadata & Prompts (IN PROGRESS)
- [ ] Add retrieval metadata to Function 2:
  - `region_name` - exact region queried
  - `ipc_phase` - from IPC dataset
  - `retrieved_chunk_count` - total chunks retrieved
  - `chunks_mentioning_region_count` - how many mention region by name
  - `retrieval_warning_flag` - boolean for retrieval quality
  - `validated_shocks` - structured list from shock detection
  - `livelihood_system` - from domain knowledge
  - `seasonal_context` - from seasonal calendar library

- [ ] Create "retrieval metadata template" that builds prompt with explicit uncertainty disclosure

- [ ] Rewrite Function 2 prompt sections:
  - A. Overview - include retrieval quality note
  - B. Livelihood System - use domain knowledge library
  - C. Seasonal Calendar - use seasonal profile, forbid invented seasons
  - D. Shocks - ONLY validated shocks from detection
  - E. Livelihood Impacts - use impact templates from library
  - F. Food Access & Consumption - FEWS-consistent coping
  - G. Nutrition & Health - only if explicitly in FEWS
  - H. IPC Alignment - use IPC definition, never escalate beyond phase
  - I. Limitations - ALWAYS required, data-driven from retrieval metadata

### Phase 3: Function 3 Driver Mapping & Prompts
- [ ] Normalize shock-to-driver mapping:
  - Check `driver_interventions.json` keys
  - Create normalized mapper from shock types to intervention JSON keys
  - Ensure "economic shocks" map to valid "economic" key (or whatever is in JSON)

- [ ] Rewrite Function 3 prompt to:
  - Accept structured driver list (pre-validated)
  - Reference specific intervention blocks by driver
  - Use Sphere/CALP/LEGS guidance by intervention type
  - Never invent new driver categories

- [ ] Fix Function 3 sections:
  - Summary - restate IPC Phase, main driver, livelihood zone
  - Immediate Emergency Actions - IPC Phase 4 emergency food & cash
  - Food & Nutrition - CMAM, blanket feeding, Sphere standards
  - Cash & Markets - CVA, vouchers, trader support (tied to economic driver)
  - Livelihood Protection - LEGS/FAO for livestock, agricultural support
  - WASH & Health - Sphere WASH and health linkages
  - Coordination - cluster-level
  - Medium-Term Recovery - resilience & early recovery
  - Limitations - note that recommendations are generic; local tailoring needed

### Phase 4: Code Updates
- [ ] Update `function2_explain_why` to:
  - Compute retrieval quality metadata
  - Pass metadata into prompt
  - Use seasonal calendar library
  - Use livelihood impact templates
  - Enforce limitations based on retrieval quality
  - Remove invented content

- [ ] Update `function3_recommend_interventions` to:
  - Normalize drivers before interventions lookup
  - Log if driver not found (to catch mapping issues)
  - Pass structured driver list to prompt
  - Use intervention library by driver

- [ ] Add new helper methods:
  - `compute_retrieval_metadata()` - returns all retrieval quality metrics
  - `normalize_drivers_for_interventions()` - maps shocks to intervention JSON keys
  - `build_limitations_statement()` - creates limitations text based on metadata

### Phase 5: Testing & Validation
- [ ] Test on EDAGA ARBI:
  - Seasonal calendar: Correct Belg/Kiremt narrative (no misshaping)
  - Shocks: Only economic shocks with evidence
  - Impacts: Only FEWS-consistent (purchasing power, coping, no invented mechanisms)
  - IPC Alignment: "Emergency (Phase 4)", no famine language
  - Limitations: Explicit about retrieval limits
  - Function 3: Economic driver maps successfully, no "Driver not mapped" warning
  - Interventions: Correctly reference CVA, market support, food assistance

- [ ] Test on other regions to ensure generalization

## Files Modified
- `src/fews_knowledge_library.py` - NEW
- `src/fews_system.py` - Updated imports, Function 2 & 3 logic
- `src/config.py` - May need new constants

## Expected Outcomes After Refactoring

### Current (Broken) Edaga Arbi Output
```
C. Seasonal Calendar
Kiremt (Secondary: Belg) - main cropping season...

D. Shocks
- Weather (generic list, possibly invented shocks)

E. Livelihood Impacts
- "Yield reduction due to economic factors" [WRONG - prices don't reduce yields]
- "Harvest timing affected by prices" [WRONG]

H. IPC Alignment
- Edaga Arbi not aligned with any specific IPC phase [CONTRADICTS IPC Phase 4]

I. Limitations
- There are no limitations reported [WRONG - should note retrieval quality]

Function 3 Log
- WARNING - Driver not mapped to shock type
```

### After Refactoring (Correct)
```
C. Seasonal Calendar
The main rainy season is June–September (kiremt), followed by meher harvest in September–October. 
The main lean season typically aligns with kiremt when food stocks are low. 
Secondary rainfall (belg) occurs in February–May and helps begin the agricultural season.

D. Shocks
- ECONOMIC: High food prices
  Evidence: "Persistently high staple food prices limiting household food access"
  Confidence: high

- ECONOMIC: Inflation/currency depreciation
  Evidence: "Wage income does not keep pace with inflation"
  Confidence: medium

E. Livelihood Impacts
Households face reduced purchasing power as wage income does not keep pace with high staple prices. 
They reduce meal size and frequency, increase reliance on market purchases, and engage in coping strategies 
such as selling livestock, sending children to relatives, or risky migration.

H. IPC Alignment
The combination of limited own-produced food stocks and high prices creates a Phase 4 (Emergency) 
level food access deficit, where households exhaust coping capacity and rely on external assistance.

I. Limitations
No documents directly mention Edaga Arbi by name. This analysis is based on zone-level FEWS NET 
reporting for northern cropping areas (Tigray and Amhara highlands) and applies general northern 
cropping patterns to this specific woreda. Local conditions and shocks may vary. Recommendation: 
conduct ground-truth verification.

Function 3 Log
✅ Driver "economic" successfully mapped to intervention bundle
  - CVA (cash-based transfer)
  - Market support (supply chain, trader credit)
  - Food assistance backup
```

## Implementation Order
1. ✅ Phase 1: Create `fews_knowledge_library.py`
2. → Phase 2: Update Function 2 prompts & metadata
3. → Phase 3: Update Function 3 driver mapping & prompts
4. → Phase 4: Code updates in `fews_system.py`
5. → Phase 5: Testing & validation on EDAGA ARBI

---

**Status**: Starting Phase 2

