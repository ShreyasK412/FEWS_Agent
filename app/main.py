import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import Filter, FieldCondition, MatchValue

import duckdb

load_dotenv()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
DUCKDB_PATH = os.getenv("DUCKDB_PATH", "fw.duckdb")
COUNTRY = os.getenv("COUNTRY", "ETH")

app = FastAPI(title="FamineWatch RAG")

_client = None
_model = None
_con = None

def qd():
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client

def emb():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def db():
    global _con
    if _con is None:
        _con = duckdb.connect(DUCKDB_PATH)
    return _con

class Passage(BaseModel):
    text: str
    source: Optional[str]=None
    round_date: Optional[str]=None
    file: Optional[str]=None

@app.get("/healthz")
def healthz():
    return {"status":"ok"}

@app.get("/search-text", response_model=List[Passage])
def search_text(q: str, k: int=6, country: str=COUNTRY, doctype: Optional[str]=None):
    vec = emb().encode(q, normalize_embeddings=True).tolist()
    flt = [FieldCondition(key="country", match=MatchValue(value=country))]
    if doctype:
        flt.append(FieldCondition(key="doctype", match=MatchValue(value=doctype)))
    hits = qd().search(
        collection_name="fs_docs",
        query_vector=vec,
        limit=k,
        query_filter={"must":flt}
    )
    return [Passage(text=h.payload.get("text",""),
                    source=h.payload.get("source"),
                    round_date=h.payload.get("round_date"),
                    file=h.payload.get("file")) for h in hits]

@app.get("/prices")
def prices(admin2: str, commodity: str, start: str, end: str):
    q = """    SELECT * FROM prices
    WHERE admin2 = ? AND commodity = ? AND month BETWEEN ? AND ?
    ORDER BY month
    """
    return db().execute(q, [admin2, commodity, start, end]).fetchall()
