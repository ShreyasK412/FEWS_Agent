# Complete Setup and Running Guide

## Prerequisites

- macOS (you're on a MacBook)
- Python 3.8 or higher
- Terminal access
- At least 8GB RAM (16GB recommended for better performance)
- Internet connection (for initial setup)

---

## Step 1: Install Ollama

Ollama is the local LLM server that will run the AI model on your MacBook.

### Option A: Using Homebrew (Recommended)
```bash
brew install ollama
```

### Option B: Manual Installation
1. Visit https://ollama.ai/download
2. Download the macOS installer
3. Open the `.dmg` file and follow the installation instructions

**Verify installation:**
```bash
ollama --version
```

---

## Step 2: Start Ollama Service

Open a **new terminal window** and keep it running. This will start the Ollama server:

```bash
ollama serve
```

You should see output like:
```
2024/01/01 12:00:00 routes.go:1008: INFO server config env="map[OLLAMA_HOST:0.0.0.0:11434]"
2024/01/01 12:00:00 routes.go:1011: INFO starting server...
```

**Keep this terminal window open** - the Ollama server needs to keep running.

---

## Step 3: Download a Language Model

Open a **new terminal window** (keep the Ollama serve window open) and run:

### Recommended: llama3.2 (Good balance of performance and size)
```bash
ollama pull llama3.2
```

This will download ~2GB. The first time may take a few minutes.

### Alternative Models (if you have less RAM or want faster responses):

**Smaller/Faster (but less capable):**
```bash
ollama pull llama3.2:1b    # ~1.3GB, very fast
ollama pull llama3.2:3b    # ~2GB, faster
```

**Larger/More Capable (slower, needs more RAM):**
```bash
ollama pull llama3.2:13b   # ~7.3GB, more capable
ollama pull llama3.1:8b    # ~4.7GB, good balance
```

**Verify the model downloaded:**
```bash
ollama list
```

You should see your model listed.

---

## Step 4: Set Up Python Environment

### Check Python Version
```bash
python3 --version
```

You need Python 3.8 or higher. If you don't have it, install via Homebrew:
```bash
brew install python3
```

### Navigate to Project Directory
```bash
cd /Users/shreyaskamath/FEWS
```

### Install Python Dependencies

**Option A: Using pip (recommended)**
```bash
pip3 install -r requirements.txt
```

**Option B: Using a virtual environment (recommended for isolation)**
```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Step 5: Verify Your Data Files

Make sure you have these files in your `/Users/shreyaskamath/FEWS` directory:
- PDF files (e.g., `0820_002-ebook.pdf`, `4409_002-ebook.pdf`, etc.)
- CSV file: `wfp_food_prices_eth (1).csv`

You can check with:
```bash
ls -la *.pdf
ls -la *.csv
```

---

## Step 6: Run the System

### Make sure:
1. ✅ Ollama is running (`ollama serve` in one terminal)
2. ✅ A model is downloaded (`ollama list` shows your model)
3. ✅ Python dependencies are installed
4. ✅ You're in the project directory

### Start the application:
```bash
python3 main.py
```

Or if you're using a virtual environment:
```bash
source venv/bin/activate
python main.py
```

---

## Step 7: First Run - What to Expect

The first time you run the system, it will:

1. **Check Ollama connection** - Verifies Ollama is running
2. **Process PDFs** - Extracts text from all PDF files (this may take a few minutes)
3. **Process CSV** - Analyzes price data and creates summaries
4. **Create vector database** - Builds embeddings (this may take 5-10 minutes the first time)
5. **Start interactive interface** - You'll see a prompt to ask questions

Example output:
```
================================================================================
FOOD INSECURITY ANALYSIS SYSTEM FOR ETHIOPIA
================================================================================

Initializing system...

Step 1: Processing documents...
Processing PDF: 0820_002-ebook.pdf
  Extracted 45 chunks from 120 pages
Processing PDF: 4409_002-ebook.pdf
  Extracted 32 chunks from 85 pages
...
Processing CSV: wfp_food_prices_eth (1).csv
  Created 45 chunks from CSV data

Total chunks created: 234

Step 2: Initializing RAG system with model 'llama3.2'...
✓ Embeddings initialized with model: llama3.2
✓ LLM initialized with model: llama3.2

Step 3: Creating vector store...
Creating new vector store...
✓ Vector store created
Vector store contains 234 documents

Step 4: Setting up QA chain...
✓ QA chain setup complete

✓ System ready!

================================================================================
FOOD INSECURITY ANALYSIS SYSTEM FOR ETHIOPIA
================================================================================

Ask questions about food insecurity in Ethiopia.
The system will consult your documents and price data.
...
```

---

## Step 8: Ask Questions

Once the system is ready, you can ask questions. Examples:

```
Question: What are the main factors affecting food insecurity in Ethiopia?
```

```
Question: What are the current maize prices in Addis Ababa?
```

```
Question: How has the price of Teff changed over time?
```

```
Question: What regions in Ethiopia have the highest food prices?
```

```
Question: What do the documents say about food security interventions?
```

### Commands:
- Type your question and press Enter
- Type `help` for example questions
- Type `exit` or `quit` to exit

---

## Troubleshooting

### Problem: "Error connecting to Ollama"

**Solution:**
1. Make sure Ollama is running: Open a terminal and run `ollama serve`
2. Check if Ollama is accessible: `curl http://localhost:11434/api/tags`
3. If you get a connection error, restart Ollama: Press Ctrl+C in the Ollama serve window, then run `ollama serve` again

### Problem: "No models found" or "Model not found"

**Solution:**
1. Check available models: `ollama list`
2. If empty, download a model: `ollama pull llama3.2`
3. If you get an error, try: `ollama pull llama3.2:1b` (smaller model)

### Problem: "Module not found" errors

**Solution:**
1. Make sure you're in the project directory: `cd /Users/shreyaskamath/FEWS`
2. Reinstall dependencies: `pip3 install -r requirements.txt`
3. If using virtual environment, make sure it's activated: `source venv/bin/activate`

### Problem: PDFs not processing

**Solution:**
1. Check if PDFs are in the correct directory: `ls *.pdf`
2. Make sure you have read permissions
3. Check if PDFs are corrupted by trying to open one manually

### Problem: Slow processing

**This is normal on first run!** The system needs to:
- Extract text from PDFs (can take 1-5 minutes depending on PDF size)
- Create embeddings (can take 5-15 minutes depending on data size)
- Subsequent runs will be much faster as the vector database is cached

### Problem: Out of memory errors

**Solution:**
1. Use a smaller model: `ollama pull llama3.2:1b` or `llama3.2:3b`
2. Close other applications to free up RAM
3. Edit `main.py` and change `MODEL_NAME = "llama3.2"` to `MODEL_NAME = "llama3.2:1b"`

### Problem: ChromaDB errors

**Solution:**
1. Delete the vector database: `rm -rf chroma_db`
2. Run the system again - it will recreate the database

---

## Performance Tips

1. **First run is slow** - The system processes all documents and creates embeddings. This is normal and only happens once (or when you add new documents).

2. **Subsequent runs are fast** - The vector database is cached in `chroma_db/`, so future runs start immediately.

3. **Model size matters** - Larger models (13b, 8b) give better answers but are slower and need more RAM. Smaller models (1b, 3b) are faster but less capable.

4. **Query speed** - Each query takes 5-30 seconds depending on:
   - Model size
   - Query complexity
   - Number of documents retrieved

---

## Adding New Documents

To add new documents:

1. Place new PDF or CSV files in the `/Users/shreyaskamath/FEWS` directory
2. Delete the old vector database: `rm -rf chroma_db`
3. Run `python3 main.py` again - it will process all documents including the new ones

---

## Stopping the System

1. Type `exit` or `quit` in the query interface
2. Press Ctrl+C if needed
3. Stop Ollama by pressing Ctrl+C in the terminal running `ollama serve`

---

## Quick Reference Commands

```bash
# Start Ollama
ollama serve

# Download model (in another terminal)
ollama pull llama3.2

# List models
ollama list

# Run the system
cd /Users/shreyaskamath/FEWS
python3 main.py

# Check if Ollama is running
curl http://localhost:11434/api/tags

# Reset vector database (if needed)
rm -rf chroma_db
```

---

## Next Steps

Once everything is running:

1. Try different types of questions
2. Experiment with different models if you want better/faster responses
3. Add more documents to expand the knowledge base
4. Customize the prompt in `rag_system.py` if you want different answer styles

---

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Verify all prerequisites are installed
3. Check that Ollama is running and a model is downloaded
4. Review the error messages - they usually indicate what's wrong

Good luck with your food insecurity analysis!

