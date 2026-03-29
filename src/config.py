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

MAX_QUESTIONS_PER_TAG = 100
MIN_ANSWER_SCORE = 1  

GITHUB_REPOS = [
    "ultralytics/ultralytics",
    "pytorch/pytorch",
    "pytorch/vision",
]
MAX_ISSUES_PER_REPO = 200
GITHUB_TOKEN = env  

# ===== Data Processing =====
MIN_ERROR_LENGTH = 20       
MAX_ERROR_LENGTH = 3000     
MIN_FIX_LENGTH = 30         

