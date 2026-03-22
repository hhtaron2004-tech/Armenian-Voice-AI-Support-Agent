"""
store.py — Data loading, chunking, embedding, and ChromaDB storage.
Combines chunker, embedder, and vector_store into one module.
"""

import json
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from src.config import DATA_DIR, EMBEDDINGS_DIR, SUPPORTED_BANKS, CHROMA_COLLECTION_NAME, EMBEDDING_MODEL

# ─── Embedding Model ───────────────────────────────────────────────

_model = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[store] Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model

def embed_texts(texts: List[str]) -> List[List[float]]:
    return get_model().encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

def embed_query(query: str) -> List[float]:
    return get_model().encode(query, convert_to_numpy=True).tolist()

# ─── JSON Loading ──────────────────────────────────────────────────

def load_json(filepath: Path) -> Any:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

# ─── Chunking ──────────────────────────────────────────────────────

def chunk_deposit(record: Dict) -> Dict[str, Any]:
    content = record.get("content", "")
    meta = record.get("metadata", {})
    return {
        "id": record["id"],
        "text": content,
        "bank": meta.get("bank", ""),
        "bank_id": meta.get("bank_id", ""),
        "type": "deposit",
        "product": meta.get("product", ""),
        "topic": meta.get("topic", ""),
        "source_url": meta.get("source_url", ""),
    }

def chunk_credit(record: Dict) -> List[Dict[str, Any]]:
    chunks = []
    bank = record.get("bank", "")
    bank_id = bank.lower().replace(" ", "_")
    product = record.get("product", "")
    source_url = record.get("source_url", "")
    topics = record.get("topics", {})
    record_id = record.get("id", f"{bank_id}_{product}")
    for topic_key, topic_text in topics.items():
        chunks.append({
            "id": f"{record_id}_{topic_key}",
            "text": topic_text,
            "bank": bank,
            "bank_id": bank_id,
            "type": "credit",
            "product": product,
            "topic": topic_key,
            "source_url": source_url,
        })
    return chunks

def chunk_branch(record: Dict) -> Dict[str, Any]:
    meta = record.get("metadata", {})
    return {
        "id": record["id"],
        "text": record.get("content", ""),
        "bank": meta.get("bank", ""),
        "bank_id": meta.get("bank", "").lower().replace(" ", "_"),
        "type": "branch",
        "product": "branch",
        "topic": "location",
        "city": meta.get("city", ""),
        "region": meta.get("region", ""),
        "address": meta.get("address", ""),
        "working_hours": meta.get("working_hours", ""),
        "source_url": meta.get("source_url", ""),
    }

def load_all_chunks() -> List[Dict[str, Any]]:
    all_chunks = []
    for bank_id in SUPPORTED_BANKS:
        bank_dir = DATA_DIR / bank_id
        if not bank_dir.exists():
            print(f"[store] Warning: {bank_dir} not found, skipping.")
            continue
        deposit_file = bank_dir / "deposits.json"
        if deposit_file.exists():
            for record in load_json(deposit_file):
                all_chunks.append(chunk_deposit(record))
        credit_file = bank_dir / "credits.json"
        if credit_file.exists():
            credit_data = load_json(credit_file)
            records = credit_data.get("credits", []) if isinstance(credit_data, dict) else credit_data
            for record in records:
                all_chunks.extend(chunk_credit(record))
        branch_file = bank_dir / "branches.json"
        if branch_file.exists():
            for record in load_json(branch_file):
                all_chunks.append(chunk_branch(record))
    print(f"[store] Loaded {len(all_chunks)} chunks from {len(SUPPORTED_BANKS)} banks.")
    return all_chunks

# ─── ChromaDB ──────────────────────────────────────────────────────

def get_collection():
    client = chromadb.PersistentClient(path=str(EMBEDDINGS_DIR))
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def build_vector_store(chunks: List[Dict[str, Any]]) -> None:
    collection = get_collection()
    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "bank": c.get("bank", ""),
            "bank_id": c.get("bank_id", ""),
            "type": c.get("type", ""),
            "product": c.get("product", ""),
            "topic": c.get("topic", ""),
            "source_url": c.get("source_url", ""),
            "city": c.get("city", ""),
            "region": c.get("region", ""),
        }
        for c in chunks
    ]
    print(f"[store] Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_texts(texts)
    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size],
            documents=texts[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )
        print(f"[store] Upserted batch {i//batch_size + 1}")
    print(f"[store] Done. {collection.count()} documents in collection.")

def query_vector_store(
    query: str,
    top_k: int = 5,
    filter_bank: Optional[str] = None,
    filter_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    collection = get_collection()
    query_emb = embed_query(query)
    where_filter = {}
    if filter_bank and filter_type:
        where_filter = {"$and": [{"bank_id": filter_bank}, {"type": filter_type}]}
    elif filter_bank:
        where_filter = {"bank_id": filter_bank}
    elif filter_type:
        where_filter = {"type": filter_type}
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k,
        where=where_filter if where_filter else None,
        include=["documents", "metadatas", "distances"],
    )
    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({"text": doc, "metadata": meta, "score": 1 - dist})
    return output