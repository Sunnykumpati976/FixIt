import json
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, FAISS_INDEX_PATH

PROCESSED_DIR = Path("data/processed")

DOC_URLS = [
    "https://pytorch.org/docs/stable/notes/cuda.html",
    "https://pytorch.org/docs/stable/notes/faq.html",
    "https://pytorch.org/docs/stable/notes/multiprocessing.html",
    "https://docs.ultralytics.com/guides/yolo-common-issues/",
    "https://docs.ultralytics.com/quickstart/",
    "https://pytorch.org/docs/stable/onnx.html",
    "https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html",
]


def download_docs(urls: list[str]) -> list[dict]:
    """Fetch documentation pages and extract plain text.

    Args:
        urls: List of documentation URLs to fetch.

    Returns:
        List of {"url": "...", "text": "..."} dicts.
    """
    docs = []
    for url in urls:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            soup = BeautifulSoup(resp.text, "html.parser")

            for tag in soup.find_all(["nav", "footer", "script", "style", "header"]):
                tag.decompose()

            text = soup.get_text(separator="\n")

            lines = [line.strip() for line in text.splitlines()]
            text = "\n".join(line for line in lines if line)

            if len(text) > 100:
                docs.append({"url": url, "text": text})
                print(f"  [✓] {url} ({len(text)} chars)")
            else:
                print(f"  [!] {url} — too little content, skipping")

        except requests.RequestException as e:
            print(f"  [!] {url} — failed: {e}")

    return docs


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The full document text.
        chunk_size: Max characters per chunk.
        overlap: Characters of overlap between adjacent chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be < chunk_size")

    chunks = []
    start = 0
    step = chunk_size - overlap

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start += step

    return chunks


def build_index():
    """Main pipeline: download → chunk → embed → save FAISS index."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("[*] Downloading documentation...")
    docs = download_docs(DOC_URLS)
    print(f"    Downloaded {len(docs)} pages")

    if not docs:
        print("[!] No docs downloaded. Check your internet connection.")
        return

    print("[*] Chunking documents...")
    all_chunks = []

    for doc in docs:
        chunks = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "url": doc["url"],
                "chunk_idx": i,
            })

    print(f"    Created {len(all_chunks)} chunks from {len(docs)} docs")

    print("[*] Embedding chunks (this may take a minute)...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    chunk_texts = [c["text"] for c in all_chunks]
    embeddings = model.encode(chunk_texts, show_progress_bar=True)

    print(f"    Embedded {len(embeddings)} chunks → shape {embeddings.shape}")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    print(f"    FAISS index built: {index.ntotal} vectors, {dimension} dimensions")

    faiss.write_index(index, str(PROCESSED_DIR / "faiss_index.bin"))
    with open(PROCESSED_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\n[✓] RAG index saved!")
    print(f"    Index: {PROCESSED_DIR / 'faiss_index.bin'}")
    print(f"    Chunks: {PROCESSED_DIR / 'chunks.json'}")
    print(f"    Total chunks indexed: {len(all_chunks)}")


if __name__ == "__main__":
    build_index()