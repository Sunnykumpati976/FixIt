"""
Central configuration for the FixIt project.
All magic numbers, API endpoints, and model choices live here.
"""

# ===== Data Collection =====
STACKOVERFLOW_API = "https://api.stackexchange.com/2.3"
STACKOVERFLOW_TAGS = [
    "pytorch",
    "yolov5",
    "yolov8",
    "ultralytics",
    "cuda",
    "torchvision",
    "torch",
    "onnx",
    "tensorrt",
]

MAX_QUESTIONS_PER_TAG = 200
MIN_ANSWER_SCORE = 1  

GITHUB_REPOS = [
    "ultralytics/ultralytics",
    "pytorch/pytorch",
    "pytorch/vision",
    "huggingface/transformers",
    "open-mmlab/mmdetection",
]
MAX_ISSUES_PER_REPO = 500

# ===== Data Processing =====
MIN_ERROR_LENGTH = 20       
MAX_ERROR_LENGTH = 3000     
MIN_FIX_LENGTH = 30         

# ===== RAG Pipeline =====
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "data/processed/faiss_index"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5