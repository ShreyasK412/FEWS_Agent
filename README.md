# Food Insecurity Analysis System for Ethiopia

A self-hosted RAG (Retrieval Augmented Generation) system that uses a local LLM to analyze food insecurity in Ethiopia by consulting your documents and price data.

## Quick Start

1. **Install Ollama** (if not installed):
   ```bash
   brew install ollama
   ```

2. **Start Ollama** (keep this terminal open):
   ```bash
   ollama serve
   ```

3. **Download a model** (in a new terminal):
   ```bash
   ollama pull llama3.2
   ```

4. **Install Python dependencies**:
   ```bash
   cd /Users/shreyaskamath/FEWS
   pip3 install -r requirements.txt
   ```

5. **Run the system**:
   ```bash
   python3 main.py
   ```

## Detailed Setup Guide

📖 **For complete step-by-step instructions, troubleshooting, and tips, see [SETUP_GUIDE.md](SETUP_GUIDE.md)**

## Usage

The system will:
1. Process all PDF documents in the directory
2. Process the CSV price data
3. Create a vector database for fast retrieval
4. Start an interactive query interface

You can ask questions like:
- "What are the main factors affecting food insecurity in Ethiopia?"
- "What are the current maize prices in Addis Ababa?"
- "How has the price of Teff changed over time?"
- "What do the documents say about food security interventions?"

## Project Structure

- `main.py` - Main application entry point
- `document_processor.py` - Handles PDF and CSV processing
- `rag_system.py` - RAG system with vector store and retrieval
- `query_interface.py` - Interactive query interface
- `SETUP_GUIDE.md` - Complete setup and troubleshooting guide
- `requirements.txt` - Python dependencies

