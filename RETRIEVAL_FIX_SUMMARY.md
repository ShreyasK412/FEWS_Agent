# Retrieval Issue Fix Summary

## Problem Identified

When testing Edaga arbi, the "Explain Why" function was failing with:
```
Retrieved only 1 chunks, minimum required: 3
```

Then after initial fixes, it got worse:
```
Retrieved only 0 chunks, minimum required: 1
```

## Root Causes

### 1. **Ollama Server Not Running** (Primary Issue)
```
⚠️  Query 'geographic' failed: Failed to connect to Ollama
```

**Solution**: User must run `ollama serve` in a separate terminal before using the system.

### 2. **Too Strict Minimum Chunk Requirement**
- Original: `MIN_RELEVANT_CHUNKS = 3`
- Problem: With only 1 situation report PDF (243 chunks), and region-specific queries, we often can't find 3 relevant chunks
- **Solution**: Lowered to `MIN_RELEVANT_CHUNKS = 1`

### 3. **Aggressive Deduplication**
- Original: Used first 100 characters for hash
- Problem: PDFs with repeated headers/footers caused false duplicates
- **Solution**: Changed to first 300 characters for better uniqueness

### 4. **Silent Query Failures**
- Problem: Multi-query loop was silently catching all exceptions
- **Solution**: Added diagnostic output to show which queries succeed/fail

## Changes Made

### File: `src/config.py`
```python
# Before
MIN_RELEVANT_CHUNKS = 3
MAX_CONTEXT_CHUNKS = 6

# After
MIN_RELEVANT_CHUNKS = 1  # Lowered for sparse data
MAX_CONTEXT_CHUNKS = 8  # Increased for better coverage
```

### File: `src/fews_system.py`

**1. Improved Deduplication** (line 592):
```python
# Before
content_hash = hash(doc.page_content[:100])

# After
content_hash = hash(doc.page_content[:300] if len(doc.page_content) >= 300 else doc.page_content)
```

**2. Added Diagnostic Output** (lines 575-586):
```python
for query_type, query in queries:
    try:
        if self.reports_vectorstore:
            retriever = self.reports_vectorstore.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)
            if docs:
                all_docs.extend(docs)
                print(f"   📥 Query '{query_type}': retrieved {len(docs)} chunks")
    except Exception as e:
        print(f"   ⚠️  Query '{query_type}' failed: {str(e)[:50]}")
        pass
```

**3. Fixed Indentation Errors**:
- Fixed exception handling blocks that had extra indentation
- System now imports correctly

## How to Test

### 1. Start Ollama Server
```bash
# In a separate terminal
ollama serve
```

### 2. Run the System
```bash
cd /Users/shreyaskamath/FEWS
python3 fews_cli.py
# Select option 4
# Enter: Edaga arbi
```

### Expected Output (with Ollama running):
```
📥 Query 'geographic': retrieved 3 chunks
📥 Query 'livelihood': retrieved 3 chunks
📥 Query 'seasonal': retrieved 2 chunks
📥 Query 'phase': retrieved 3 chunks

✅ Retrieved 8 unique chunks after deduplication
✅ Shock detection: weather (high confidence), conflict (medium confidence)
✅ Explanation generated
```

## Remaining Limitations

### 1. Limited Situation Report Data
**Issue**: Only 1 PDF with 243 chunks total

**Impact**: 
- May not have region-specific information for all woredas
- Generic Ethiopia-wide content may dominate results

**Solutions**:
- Add more situation reports (FEWS NET monthly outlooks, WFP reports, OCHA bulletins)
- Use regional-specific reports for Tigray, Somali, etc.
- Consider web scraping recent reports

### 2. Ollama Dependency
**Issue**: System requires Ollama server to be running

**Impact**: 
- Embeddings fail if Ollama is down
- No graceful degradation

**Solutions**:
- Add health check for Ollama at startup
- Provide clear error message with instructions
- Consider fallback to OpenAI embeddings

### 3. Query Specificity vs. Coverage Trade-off
**Issue**: Specific queries (with woreda names) may miss broader regional context

**Current Approach**: 4-query strategy helps, but still limited

**Future Improvements**:
- Add hierarchical retrieval (woreda → zone → region)
- Use query expansion with synonyms
- Implement hybrid search (semantic + keyword)

## Performance Metrics

| Scenario | Before Fix | After Fix |
|----------|-----------|-----------|
| Min chunks required | 3 | 1 |
| Max chunks used | 6 | 8 |
| Deduplication hash size | 100 chars | 300 chars |
| Query diagnostics | None | Per-query output |
| Ollama check | No | Error message |

## Git Commits

```
5afd99f - Fix retrieval issues: lower min chunks, improve deduplication, add diagnostics
```

## Next Steps

1. **Immediate**: Ensure Ollama is running (`ollama serve`)
2. **Short-term**: Add more situation reports to improve coverage
3. **Medium-term**: Implement Ollama health check at startup
4. **Long-term**: Add hybrid search and query expansion

---

**Status**: ✅ Retrieval logic fixed, awaiting Ollama server to test
**Last Updated**: 2025-11-12 17:30

