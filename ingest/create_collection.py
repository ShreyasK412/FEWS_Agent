import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

load_dotenv()
HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = int(os.getenv("QDRANT_PORT", "6333"))

client = QdrantClient(host=HOST, port=PORT)
client.recreate_collection(
    collection_name="fs_docs",
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)
)
print(f"Collection 'fs_docs' ready on {HOST}:{PORT}.")
