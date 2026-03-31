"""
Retrieve relevant documentation chunks for a given error.

Loads the FAISS index + chunk metadata saved by build_index.py,
embeds the user's error query, and returns the top-K most similar
doc chunks.

Usage:
    retriever = Retriever()
    results = retriever.retrieve("RuntimeError: CUDA out of memory...")
    for chunk in results:
        print(chunk["text"], chunk["url"])
"""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, TOP_K_RETRIEVAL

PROCESSED_DIR = Path("data/processed")


class Retriever:
    """Loads the FAISS index and retrieves relevant doc chunks.

    How it works:
      1. __init__ loads the saved FAISS index and chunk metadata
      2. retrieve() embeds your query into the SAME vector space
      3. FAISS finds the closest vectors (= most similar doc chunks)
      4. Returns the actual text of those chunks + their source URLs
    """

    def __init__(self):
        """Load the FAISS index, chunk metadata, and embedding model."""
        index_path = PROCESSED_DIR / "faiss_index.bin"
        chunks_path = PROCESSED_DIR / "chunks.json"

        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                "RAG index not found. Run: python -m src.rag.build_index"
            )

        # Load the FAISS index (the vectors)
        self.index = faiss.read_index(str(index_path))

        # Load the chunk metadata (the actual text + URLs)
        with open(chunks_path) as f:
            self.chunks = json.load(f)

        # Load the SAME embedding model used to build the index
        # This is critical — query and docs must be in the same vector space
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        print(f"[✓] Retriever loaded: {self.index.ntotal} chunks indexed")

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Find the most relevant doc chunks for a query.

        Args:
            query: The error message or traceback to search for.
            top_k: Number of chunks to return.

        Returns:
            List of {"text": "...", "url": "...", "score": float} dicts,
            sorted by relevance (most relevant first).
        """
        # Step 1: Embed the query into the same vector space as the docs
        query_vector = self.model.encode([query]).astype(np.float32)

        # Step 2: Search FAISS for nearest neighbors
        # distances = L2 distances (lower = more similar)
        # indices = positions in the original chunk list
        distances, indices = self.index.search(query_vector, top_k)

        # Step 3: Map indices back to actual chunk text
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue  # FAISS returns -1 for missing results

            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "url": chunk["url"],
                "score": float(dist),  # Lower = more relevant
            })

        return results


def test_retriever():
    """Quick test to verify the retriever works."""
    retriever = Retriever()

    test_queries = [
        "RuntimeError: CUDA out of memory",
        "RuntimeError: Expected all tensors to be on the same device",
        "DataLoader worker is killed by signal",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")

        results = retriever.retrieve(query, top_k=3)
        for i, r in enumerate(results):
            print(f"\n  [{i+1}] Score: {r['score']:.4f}")
            print(f"      Source: {r['url']}")
            # Show first 200 chars of the chunk
            print(f"      Text: {r['text'][:200]}...")


if __name__ == "__main__":
    test_retriever()