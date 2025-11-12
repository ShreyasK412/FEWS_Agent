# 🚀 Quick Start: Fix Vector Store in 3 Steps

## The Problem
Your vector store was broken - it retrieved 12 chunks but ended up with only 1, and that chunk was about Borena (wrong region). This meant shock detection was getting irrelevant context.

## The Solution

### Step 1️⃣ Check Current State (2 minutes)

```bash
python3 check_vector_store.py
```

**What to look for:**
- If you see mostly ✅ RELEVANT → Skip to Step 3
- If you see mostly ❌ WRONG REGION → Do Step 2
- If mixed → Do Step 2

### Step 2️⃣ Rebuild Vector Store (if needed) (10-15 minutes)

```bash
# Delete old broken vector store
rm -rf chroma_db_reports/

# Restart - will rebuild with improved chunking
python3 fews_cli.py

# It will take 5-10 minutes. Wait for it to finish.
# You'll see: "✅ Vector store setup complete"
```

**Why rebuild?**
- New chunking (1500 chars instead of 1000) keeps regional sections together
- Better overlap (300 instead of 200) preserves context
- Deduplication uses 300 chars (not 100), so better distinguishes chunks

### Step 3️⃣ Test It Works (3 minutes)

```bash
python3 fews_cli.py
# Select: 4 (Full analysis)
# Enter: EDAGA ARBI
# Wait for results...
```

**Good signs:**
```
✅ Retrieved 12 total chunks, 8 after dedup, 6 after filtering
✅ 5/6 chunks mention Tigray Region
✅ Detected 3 shocks
  - weather: Delayed kiremt rains
  - conflict: Labor migration constrained
  - economic: Fuel shortages
✅ Evidence quotes are from Tigray, not Borena
```

**Bad signs:**
```
❌ Retrieved 12 chunks, 1 after filtering
❌ 0/1 chunks mention Tigray
❌ Evidence: "Borena-Somali border conflict"
```

If you see bad signs → Something went wrong, read VECTOR_STORE_FIXES.md

---

## What Was Actually Fixed

| Issue | Fix |
|-------|-----|
| Deduplication too aggressive | Use 300 chars instead of 100 |
| Geographic filtering too strict | Only exclude clear wrong-region content |
| PDF chunking splitting sections | Larger chunks + better overlap |
| No fallback when filtering empties | Use all chunks instead of failing |
| Manual context missing | Added Tigray/Amhara manual context files |

---

## Why This Matters

**Before:** Shock detection was getting Borena-Somali border content for Edaga Arbi (Tigray).

**After:** 
- Manual context ensures Tigray-specific information is always included
- Better chunking keeps regional sections intact
- Smarter filtering allows useful content while excluding irrelevant content
- Result: Specific, accurate shocks with proper evidence

---

## If Still Broken

1. Run `check_vector_store.py` → shows diagnosis
2. Look at logs: `missing_info.log` → shows what's being retrieved
3. See section "Troubleshooting" in `VECTOR_STORE_FIXES.md`

That's it! Vector store should be working now. 🎉

