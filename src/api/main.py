"""
FixIt API — ML Error Diagnosis Endpoint.

Uses RAG retrieval + Gemini API for diagnosis generation.
Works on any machine, no GPU needed.

Open http://localhost:8000 for the web UI.
"""

import os
import time
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.rag.retriever import Retriever

# ============================================================
# Config
# ============================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
MAX_INPUT_LENGTH = 5000

SYSTEM_PROMPT = (
    "You are FixIt, an expert ML debugging assistant. "
    "Given a PyTorch, YOLO, or CUDA error message or traceback, "
    "diagnose the root cause and provide a clear, actionable fix. "
    "Include corrected code when applicable. "
    "Be concise and specific. Structure your response as: "
    "1) Diagnosis (what went wrong), "
    "2) Root cause, "
    "3) Fix (with code if applicable)."
)

# ============================================================
# Schemas
# ============================================================

class DiagnoseRequest(BaseModel):
    error_log: str = Field(..., description="The error message or traceback")
    top_k: int = Field(default=5, description="Number of docs to retrieve")


class Source(BaseModel):
    url: str
    score: float


class DiagnoseResponse(BaseModel):
    diagnosis: str
    sources: list[Source]
    mode: str
    latency_ms: float


# ============================================================
# App
# ============================================================

app = FastAPI(
    title="FixIt API",
    description="ML Error Diagnosis — RAG + Gemini LLM",
    version="1.0.0",
)

retriever = None


@app.on_event("startup")
async def startup():
    global retriever
    print("[*] Loading RAG retriever...")
    retriever = Retriever()
    if GEMINI_API_KEY:
        print("[✓] Gemini API key found — LLM mode enabled")
    else:
        print("[!] No GEMINI_API_KEY set — RAG-only mode")
        print("    Run: export GEMINI_API_KEY=your-key-here")


def call_gemini(error_log: str, context: str) -> str:
    """Call Gemini API with error + RAG context."""
    user_message = (
        f"Diagnose this error and provide a fix.\n\n"
        f"Error:\n```\n{error_log}\n```\n\n"
        f"Relevant documentation:\n{context}"
    )

    payload = {
        "system_instruction": {
            "parts": [{"text": SYSTEM_PROMPT}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [{"text": user_message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 1024
        }
    }

    response = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=30,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {response.status_code} — {response.text}"
        )

    data = response.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise HTTPException(
            status_code=502,
            detail=f"Unexpected Gemini response format: {data}"
        )


@app.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(request: DiagnoseRequest):
    """Diagnose an ML error and provide a fix."""
    start_time = time.time()

    if not request.error_log.strip():
        raise HTTPException(status_code=400, detail="error_log cannot be empty")
    if len(request.error_log) > MAX_INPUT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"error_log exceeds {MAX_INPUT_LENGTH} chars"
        )

    # RAG retrieval
    rag_results = retriever.retrieve(request.error_log, top_k=request.top_k)
    context = "\n\n---\n\n".join(
        f"[Source: {r['url']}]\n{r['text']}" for r in rag_results
    )

    # Generate diagnosis
    if GEMINI_API_KEY:
        diagnosis = call_gemini(request.error_log, context)
        mode = "rag+gemini"
    else:
        diagnosis = (
            f"## Relevant Documentation\n\n{context}\n\n"
            f"---\n*Set GEMINI_API_KEY for AI-generated diagnosis.*"
        )
        mode = "rag-only"

    latency = (time.time() - start_time) * 1000

    return DiagnoseResponse(
        diagnosis=diagnosis,
        sources=[Source(url=r["url"], score=r["score"]) for r in rag_results],
        mode=mode,
        latency_ms=round(latency, 1),
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rag_loaded": retriever is not None,
        "llm_enabled": GEMINI_API_KEY is not None,
        "provider": "gemini" if GEMINI_API_KEY else "none",
    }


# Serve the frontend
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    @app.get("/")
    async def root():
        return FileResponse(STATIC_DIR / "index.html")