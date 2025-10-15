import os, json, uuid, pickle
from pathlib import Path
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Chroma
import chromadb
from chromadb.config import Settings

# FAISS
import faiss
import numpy as np

load_dotenv()
DATA_DIR = Path("data")
INDEX_DIR = Path("indexes")
INDEX_DIR.mkdir(exist_ok=True)

def load_docs() -> List[Dict]:
    docs = []
    for f in DATA_DIR.glob("*.json"):
        obj = json.loads(f.read_text(encoding="utf-8"))
        md = obj.get("markdown", "")
        if not md.strip():
            continue
        docs.append({
            "url": obj.get("url", ""),
            "title": obj.get("title", ""),
            "text": md
        })
    return docs

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    # Simple character-based chunking (works well for Markdown)
    chunks = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + chunk_size, n)
        chunk = text[i:end]
        chunks.append(chunk)
        i = end - overlap if (end - overlap) > i else end
    return [c for c in chunks if c.strip()]

def build_embeddings(texts: List[str], model_name: str) -> np.ndarray:
    encoder = SentenceTransformer(model_name)
    embs = encoder.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs

def build_chroma(chunks, embeddings, metadatas, collection_name="usek"):
    db_path = str(INDEX_DIR / "chroma")
    client = chromadb.PersistentClient(path=db_path, settings=Settings(allow_reset=True))
    # recreate the collection fresh
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    col = client.create_collection(collection_name)
    ids = [str(uuid.uuid4()) for _ in chunks]
    col.add(ids=ids, documents=chunks, metadatas=metadatas, embeddings=embeddings.tolist())
    print(f"[Chroma] Indexed {len(chunks)} chunks at {db_path}")

def build_faiss(chunks, embeddings, metadatas):
    # L2 index; embeddings are normalized so inner product ≈ cosine
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
    with open(INDEX_DIR / "faiss_meta.pkl", "wb") as f:
        pickle.dump({"chunks": chunks, "metadatas": metadatas}, f)
    print(f"[FAISS] Indexed {len(chunks)} chunks at indexes/faiss.index")

def main():
    model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    store = os.getenv("STORE", "chroma")  # default build target if you run directly
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    docs = load_docs()
    if not docs:
        print("No docs found in data/. Run crawler first.")
        return

    # explode to chunks
    all_chunks, all_meta = [], []
    for d in docs:
        parts = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for p in parts:
            all_chunks.append(p)
            all_meta.append({"source": d["url"], "title": d["title"]})

    print(f"Total chunks: {len(all_chunks)} (chunk_size={chunk_size}, overlap={overlap})")
    embs = build_embeddings(all_chunks, model_name)

    if store.lower() == "chroma":
        build_chroma(all_chunks, embs, all_meta, collection_name="usek")
    else:
        build_faiss(all_chunks, embs, all_meta)

if __name__ == "__main__":
    main()
