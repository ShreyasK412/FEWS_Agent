import os, uuid, argparse, fitz
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct

def chunk(text, size=800, overlap=120):
    words=text.split()
    i=0
    while i<len(words):
        yield " ".join(words[i:i+size])
        i += (size-overlap)

def main(path, country, source, round_date, doctype):
    load_dotenv()
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
    qhost = os.getenv("QDRANT_HOST", "localhost")
    qport = int(os.getenv("QDRANT_PORT", "6333"))

    model = SentenceTransformer(model_name)
    client = QdrantClient(host=qhost, port=qport)

    doc = fitz.open(path)
    raw = "\n".join(p.get_text("text") for p in doc)

    points=[]
    for ch in chunk(raw):
        if not ch.strip():
            continue
        emb = model.encode(ch, normalize_embeddings=True).tolist()
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={
                "text": ch,
                "country": country,
                "source": source,
                "round_date": round_date,
                "doctype": doctype,
                "file": os.path.basename(path)
            }
        ))
    client.upsert(collection_name="fs_docs", points=points)
    print(f"Ingested {len(points)} chunks from {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--country", default="ETH")
    ap.add_argument("--source", default="IPC")
    ap.add_argument("--round_date", default="unknown")
    ap.add_argument("--doctype", default="bulletin")
    args = ap.parse_args()
    main(args.pdf, args.country, args.source, args.round_date, args.doctype)
