# FEWS Refactoring - Detailed Implementation Instructions

**Status**: Infrastructure in place (commit fc9e532)  
**Next**: Apply this instruction set to Cursor for full implementation

## What Has Been Done ✅
1. Created `src/fews_knowledge_library.py` with:
   - Seasonal calendars per livelihood zone (northern_cropping, pastoral_south)
   - Livelihood impact templates based on FEWS NET language
   - IPC Phase definitions  
   - Driver-to-intervention mappings

2. Created `REFACTORING_PLAN.md` with 5-phase implementation roadmap

3. Added imports to `src/fews_system.py` for the new library

## What Needs to Be Done (Use in Cursor Refactor Dialog)

### Issue 1: Fix Remaining Indentation Errors
**Status**: `src/fews_system.py` line 910 has indentation issue

**Solution**: Use Cursor's autofix to repair indentation in the fews_system.py file, particularly around:
- Lines 896-920 (Function 2 prompt/chain creation)
- Lines 1353-1369 (Function 3 exception handlers)

Run after fixing: `python3 -m py_compile src/fews_system.py`

---

### Issue 2: Implement Retrieval Metadata in Function 2

**Location**: `src/fews_system.py`, method `function2_explain_why`

**Step 1: Create metadata computation method**

Add this as a new method to `FEWSSystem` class:

```python
def compute_retrieval_metadata(self, docs, admin_region, retrieved_shocks):
    """
    Compute retrieval quality metadata for prompt construction.
    
    Returns dict with:
    - retrieved_chunk_count: int
    - chunks_mentioning_region_count: int
    - retrieval_warning_flag: bool
    - region_mention_percentage: float (0-100)
    """
    total_chunks = len(docs)
    mention_count = sum(1 for d in docs if admin_region.lower() in d.page_content.lower())
    mention_pct = (mention_count / total_chunks * 100) if total_chunks > 0 else 0
    
    return {
        "retrieved_chunk_count": total_chunks,
        "chunks_mentioning_region_count": mention_count,
        "retrieval_warning_flag": mention_pct < 50,
        "region_mention_percentage": mention_pct
    }
```

**Step 2: Pass metadata into Function 2 prompt**

Before building the prompt, compute:

```python
metadata = self.compute_retrieval_metadata(docs, admin_region, detailed_shocks)
```

**Step 3: Add metadata fields to prompt template**

Add to prompt `input_variables`:
```python
input_variables=[
    "region",
    "ipc_phase",
    "context",
    "shocks_formatted",
    "livelihood_system",
    "rainfall_season",
    "admin_region",
    "zone_name",
    "geographic_context",
    "rainfall_clarification",
    # NEW METADATA FIELDS:
    "retrieved_chunk_count",
    "chunks_mentioning_region_count",
    "retrieval_warning_flag"
]
```

**Step 4: Include metadata in prompt invocation**

Pass when calling `chain.invoke()`:
```python
explanation = chain.invoke({
    # ... existing fields ...
    "retrieved_chunk_count": metadata["retrieved_chunk_count"],
    "chunks_mentioning_region_count": metadata["chunks_mentioning_region_count"],
    "retrieval_warning_flag": str(metadata["retrieval_warning_flag"])
})
```

---

### Issue 3: Rewrite Function 2 Prompt (Section C & I)

**Add to prompt template** (replace existing C. Seasonal Calendar and I. Limitations):

```
C. Seasonal Calendar

Based on the livelihood zone's seasonal patterns:

{rainfall_season}

{rainfall_clarification}

CRITICAL: Do not invent or misname seasons. Use the zone-level patterns above.
If Kiremt (June–September), NEVER describe as "secondary" or "dry season"—it is the MAIN RAINY season.
If no FEWS document specifies a secondary season for this woreda, describe the broader pattern instead.

---

I. Limitations

Based on retrieval metadata:
- Chunks retrieved: {retrieved_chunk_count}
- Chunks mentioning this woreda specifically: {chunks_mentioning_region_count}
- Retrieval quality: {'POOR (less than half of chunks mention this region)' if retrieval_warning_flag else 'GOOD (majority of chunks mention this region)'}

{limitations_text}

Where limitations_text is:

IF retrieval_warning_flag == True:
"No documents directly mention this woreda by name. This explanation is based on zone-level FEWS NET 
reporting for {geographic_context} and inferred for {livelihood_system} livelihoods. Local conditions 
and shocks may vary. Recommendation: conduct ground-truth verification with local partners."

ELSE:
"Evidence is drawn from FEWS NET reporting for {geographic_context}. Micro-level variation within the 
woreda may not be captured. Recommendation: validate findings through field assessment."
```

---

### Issue 4: Implement Seasonal Calendar Library Integration

**Update Function 2 prompt variable construction**:

Before prompt invocation, add:

```python
from src.fews_knowledge_library import get_seasonal_calendar

# Infer livelihood zone (e.g., northern_cropping for Tigray/Amhara)
livelihood_zone = "northern_cropping" if admin_region.lower() in ["tigray", "amhara"] else "pastoral_south"

seasonal_profile = get_seasonal_calendar(livelihood_zone)
rainfall_season_text = seasonal_profile.get("zone_description", "")
```

Then pass `rainfall_season` in prompt as the correct seasonal profile (not user LLM hallucination).

---

### Issue 5: Implement Livelihood Impact Templates

**Add to Function 2 prompt**:

In section E. Livelihood Impacts, insert this instruction:

```
E. Livelihood Impacts

For each VALIDATED SHOCK from section D, describe its specific livelihood impacts using evidence from documents.

You MUST only describe impacts that:
1. Are mentioned in the retrieved FEWS NET documents, OR
2. Are logical consequences of the shock mechanisms in those documents

You MUST NOT invent new impact mechanisms. For example:
- ❌ "High prices reduce yields" (prices don't change agriculture)
- ❌ "Harvest timing affected by market disruption" (markets don't control rainfall/planting)
- ✅ "Reduced purchasing power forces household to reduce meal size/frequency"
- ✅ "Livestock prices decline, reducing herd sales income"

Reference livelihood impact library for guidance:
{livelihood_impact_examples}
```

And pass:

```python
from src.fews_knowledge_library import get_livelihood_impacts

impact_templates = get_livelihood_impacts(livelihood_zone, validated_shocks[0]["type"])
livelihood_impact_examples = "\n".join(impact_templates[:2])
```

---

### Issue 6: Fix Function 3 Driver Mapping

**Add normalization method to FEWSSystem**:

```python
def normalize_driver_for_interventions(self, shock_type):
    """
    Map shock type to normalized driver key for interventions.json lookup.
    """
    from src.fews_knowledge_library import DRIVER_SHOCK_MAPPING
    
    for driver_key, mapping in DRIVER_SHOCK_MAPPING.items():
        for keyword in mapping.get("shock_keywords", []):
            if keyword.lower() in shock_type.lower():
                return driver_key
    
    return None  # Log warning if no match
```

**Update Function 3**:

Before prompt:

```python
# Normalize drivers for intervention lookup
validated_drivers_normalized = []
for shock in detailed_shocks:
    normalized = self.normalize_driver_for_interventions(shock["description"])
    if normalized:
        validated_drivers_normalized.append(normalized)
    else:
        missing_info_logger.warning(f"Could not normalize shock: {shock['description']}")

# Pass to prompt as validated list
```

---

### Issue 7: Update Function 3 Prompt

**Add these sections**:

```
DRIVERS AND INTERVENTIONS:

You will receive a list of validated drivers that have been mapped to intervention guidance:
- economic: CVA, market support, food assistance, livelihood protection
- weather: Food assistance, agricultural inputs, water, livelihood protection  
- conflict: Food/cash assistance, protection, safe access
- displacement: Food assistance, WASH, protection, shelter

Use ONLY these drivers. Do not invent new categories.
Interventions must reference actual humanitarian standards (Sphere, CALP, LEGS, etc.).

For Edaga Arbi (Phase 4, economic shocks):
- Immediate: General Food Distribution or Cash-Based Transfer where markets function
- Cash: Multi-purpose cash assistance (CALP guidance) for purchasing power restoration
- Livelihood: Agricultural support to maintain income; selective livestock support
- WASH: Sphere WASH standards
- Nutrition: CMAM for SAM children; blanket supplementary feeding if malnourished
- Health: Linkages to basic health and disease surveillance
- Recovery: Medium-term focus on market stabilization and income restoration
```

---

### Issue 8: Add IPC Language Guards

**Add to both Function 2 and Function 3 prompts**:

```
IPC PHASE LANGUAGE:

IPC Phase {ipc_phase} = {phase_name}
Description: {phase_description}

Rules:
- DO NOT escalate language above Phase {ipc_phase}
- Only use word "famine" if Phase == 5
- For Phase 4: use "Emergency (IPC Phase 4)" only
- For Phase 3: use "Crisis (IPC Phase 3)" only
- Never say "potential for famine" unless Phase == 5

Current phase is {ipc_phase}. Do NOT mention famine, catastrophe, or starvation unless Phase == 5.
```

**Implementation in code**:

```python
from src.fews_knowledge_library import get_ipc_phase_definition

phase_def = get_ipc_phase_definition(ipc_phase)
phase_description = phase_def.get("description", "")
phase_name = phase_def.get("name", "")
```

---

## Testing Criteria After Implementation

Run full analysis on EDAGA ARBI:

```
✅ Section C: Correct Belg/Kiremt narrative (no mislabeling)
✅ Section D: Only economic shocks with evidence
✅ Section E: FEWS-consistent impacts (purchasing power, coping - no invented mechanisms)
✅ Section H: "Emergency (Phase 4)", no famine language
✅ Section I: Explicit limitations mentioning retrieval quality
✅ Function 3: Economic driver maps successfully (no warning logs)
✅ Function 3: Interventions reference CALP/Sphere/LEGS standards
```

---

## Implementation Checklist

- [ ] Fix indentation errors in src/fews_system.py
- [ ] Add `compute_retrieval_metadata()` method
- [ ] Update Function 2 prompt with metadata fields
- [ ] Rewrite Section C (Seasonal Calendar)
- [ ] Rewrite Section I (Limitations)  
- [ ] Integrate seasonal calendar library
- [ ] Integrate livelihood impact templates
- [ ] Add IPC phase language guards
- [ ] Add driver normalization method
- [ ] Update Function 3 prompt
- [ ] Test on EDAGA ARBI
- [ ] Test on other regions
- [ ] Commit all changes with descriptive messages

---

## Next Steps

1. Copy this document into Cursor's refactoring dialog
2. Use it as a step-by-step guide to implement each section
3. Test after each major change
4. Commit incrementally with clear messages
5. Run final validation on test cases

