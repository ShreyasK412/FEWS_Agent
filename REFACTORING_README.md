# FEWS Comprehensive Refactoring - README

## Overview

This refactoring ensures the FEWS system produces **100% consistent** outputs with:
- ✅ IPC Classification Dataset
- ✅ FEWS NET Ethiopia Food Security Outlook
- ✅ Humanitarian Standards (Sphere, CALP, LEGS, FAO, WFP)

**Key Problems Solved:**
- ❌ Invented seasons (e.g., "Kiremt as secondary") → ✅ Correct FEWS NET seasonal patterns
- ❌ Hallucinated shocks → ✅ Only validated shocks from structured detection
- ❌ False livelihood impacts (e.g., "prices reduce yields") → ✅ FEWS-grounded impact templates
- ❌ Famine language for Phase 4 → ✅ Strict IPC phase constraints
- ❌ No limitations stated → ✅ Data-driven uncertainty disclosure
- ❌ Driver not mapped to interventions → ✅ Normalized driver-to-JSON mapping

## What's Been Done

### ✅ Phase 1: Infrastructure Created

**Files Added:**
1. `src/fews_knowledge_library.py` - Central library with:
   - Seasonal calendars per livelihood zone (northern_cropping, pastoral_south)
   - Livelihood impact templates based on FEWS NET language
   - IPC Phase definitions to prevent hallucination
   - Driver-to-intervention mappings for Function 3

2. `REFACTORING_PLAN.md` - 5-phase implementation roadmap

3. `REFACTORING_INSTRUCTIONS.md` - Step-by-step guide for Cursor implementation

4. `src/fews_system.py` - Updated with imports for new library

**Commits:**
- `fc9e532` - Start comprehensive FEWS refactoring - add knowledge library
- `747ec4e` - Add detailed refactoring instructions for Cursor

---

## What Needs to Be Done

### ⏳ Phase 2-5: Implementation (Cursor Refactoring)

**Use these files in Cursor:**

1. **Read first**: `REFACTORING_INSTRUCTIONS.md`
   - Contains step-by-step implementation guide
   - Shows exactly what code to add
   - Includes testing criteria

2. **Reference**: `REFACTORING_PLAN.md`
   - Full 5-phase implementation roadmap
   - Expected outputs before/after
   - Timeline and dependencies

3. **Example**: `src/fews_knowledge_library.py`
   - Shows data structures for seasonal calendars
   - Shows livelihood impact templates
   - Shows IPC definitions

### Quick Reference: 8 Major Changes

1. **Fix Indentation Errors** (src/fews_system.py line 910+)
   - Use Cursor's autofix

2. **Add Retrieval Metadata to Function 2**
   - Create `compute_retrieval_metadata()` method
   - Pass metadata to prompt
   - Use in Section I (Limitations)

3. **Rewrite Function 2 Prompt - Section C (Seasonal Calendar)**
   - Use FEWS NET seasonal profiles from library
   - Forbid Kiremt/Belg misshaping
   - Never invent seasons

4. **Rewrite Function 2 Prompt - Section I (Limitations)**
   - Make ALWAYS required
   - Data-driven from retrieval metadata
   - Explicit uncertainty disclosure

5. **Integrate Seasonal Calendar Library**
   - Call `get_seasonal_calendar()` by livelihood zone
   - Use returned profile in prompt
   - Prevent LLM hallucinations

6. **Implement Livelihood Impact Templates**
   - Call `get_livelihood_impacts()` for shock type
   - Use templates to guide LLM impacts
   - Forbid invented mechanisms

7. **Fix Function 3 Driver Mapping**
   - Add `normalize_driver_for_interventions()` method
   - Map shocks to valid driver keys in interventions.json
   - Log warnings if mapping fails

8. **Update Function 3 Prompt & Add IPC Guards**
   - Only use normalized, pre-validated drivers
   - Add IPC phase language constraints (no famine for Phase 4)
   - Reference Sphere/CALP/LEGS standards

---

## Implementation Path

### For Cursor Users:
1. Open `REFACTORING_INSTRUCTIONS.md`
2. Follow Issue 1-8 sequentially
3. Test after each major change
4. Run validation on EDAGA ARBI
5. Commit incrementally

### For Direct Implementation:
1. Review `fews_knowledge_library.py` structure
2. Follow `REFACTORING_INSTRUCTIONS.md` step-by-step
3. Use `REFACTORING_PLAN.md` as reference
4. Test outputs match expected results

---

## Testing on EDAGA ARBI

After implementation, run full analysis:

```bash
python3 fews_cli.py
# Select: 4 (Full analysis)
# Enter: EDAGA ARBI
```

**Expected Output Validation:**

```
✅ Section C (Seasonal Calendar)
   - Correct Kiremt narrative (Jun-Sep = main rainy season)
   - Belg described (Feb-May = secondary)
   - No mislabeling or invention

✅ Section D (Shocks)
   - ONLY validated shocks (e.g., "Economic: High food prices")
   - Evidence quotes included
   - No invented shocks

✅ Section E (Livelihood Impacts)
   - Uses phrases like: "reduced purchasing power", "selling livestock"
   - NO phrases like: "prices reduce yields"
   - Only FEWS-consistent mechanisms

✅ Section H (IPC Alignment)
   - States: "Emergency (Phase 4)"
   - NO mention of "famine" (Phase is 4, not 5)
   - Consistent with IPC dataset value

✅ Section I (Limitations)
   - Explicitly mentions retrieval quality
   - States: "No documents directly mention this woreda"
   - Recommends ground-truth verification

✅ Function 3 (Interventions)
   - NO log warning "Driver not mapped"
   - Economic driver successfully mapped
   - Interventions reference CALP/Sphere standards
```

---

## File Locations

**Core Implementation Files:**
- `src/fews_knowledge_library.py` - ✅ Central knowledge base
- `src/fews_system.py` - ⏳ Needs Phases 2-5 updates

**Guidance Files:**
- `REFACTORING_PLAN.md` - Full 5-phase roadmap
- `REFACTORING_INSTRUCTIONS.md` - Step-by-step guide
- `REFACTORING_README.md` - This file

---

## Success Criteria

After all phases complete:

✅ All outputs are grounded in FEWS NET text, IPC data, or humanitarian standards  
✅ No invented seasons, shocks, impacts, or famine language  
✅ Seasonal calendar uses zone-specific FEWS NET patterns  
✅ Livelihood impacts reference only validated shocks  
✅ IPC Phase language is strict (no escalation above dataset value)  
✅ Limitations are always present and data-driven  
✅ Driver-to-interventions mapping works without warnings  
✅ Edaga Arbi and other regions show correct outputs  

---

## Next Step

**→ Use Cursor to implement REFACTORING_INSTRUCTIONS.md**

For questions during implementation, reference:
- `fews_knowledge_library.py` for data structures
- `REFACTORING_PLAN.md` for overview
- `REFACTORING_INSTRUCTIONS.md` for specifics

---

**Status**: Infrastructure Ready ✅ | Implementation Pending ⏳ | Validation Required 🔬

