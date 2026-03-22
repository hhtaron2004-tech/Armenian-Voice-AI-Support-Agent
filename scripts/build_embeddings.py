"""
build_embeddings.py — One-time script to build the ChromaDB vector store.
Run this before starting the agent.

Usage:
    python scripts/build_embeddings.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.store import load_all_chunks, build_vector_store
from src.config import EMBEDDINGS_DIR


def main():
    print("=" * 50)
    print("Building Armenian Bank Knowledge Base")
    print("=" * 50)

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_all_chunks()

    if not chunks:
        print("ERROR: No chunks loaded. Check that data/ directory contains bank JSON files.")
        sys.exit(1)

    build_vector_store(chunks)

    print("\n✅ Vector store built successfully!")
    print(f"   Location: {EMBEDDINGS_DIR}")
    print(f"   Total chunks: {len(chunks)}")


if __name__ == "__main__":
    main()