"""
Process raw scraped data into clean training pairs.
 
Takes JSONL from scrapers → cleans → formats → splits into train/val.
 
Output:
  data/processed/train.jsonl
  data/processed/val.jsonl
  data/processed/stats.json

"""
 
import json
import re
import random
from pathlib import Path
 
from src.config import MIN_ERROR_LENGTH, MAX_ERROR_LENGTH, MIN_FIX_LENGTH
 
RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
 
SYSTEM_PROMPT = (
    "You are FixIt, an expert ML debugging assistant. "
    "Given a PyTorch, YOLO, or CUDA error message or traceback, "
    "diagnose the root cause and provide a clear, actionable fix. "
    "Include corrected code when applicable."
)
 
INSTRUCTION = (
    "Diagnose this error and tell me exactly how to fix it. "
    "Be specific — include the root cause and corrected code if applicable."
)
 
 
def clean_error(text: str) -> str:
    """Clean and normalize an error message."""
    # Remove excessive whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
 
    # Truncate if too long (keep first and last parts)
    if len(text) > MAX_ERROR_LENGTH:
        half = MAX_ERROR_LENGTH // 2
        text = text[:half] + "\n... [truncated] ...\n" + text[-half:]
 
    # Remove file paths that are user-specific (privacy + generalization)
    text = re.sub(r"/home/\w+/", "/home/user/", text)
    text = re.sub(r"/Users/\w+/", "/Users/user/", text)
    text = re.sub(r"C:\\Users\\\w+\\", r"C:\\Users\\user\\", text)
 
    return text.strip()
 
 
def clean_fix(text: str) -> str:
    """Clean and normalize a fix/answer."""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
 
    # Remove "Edit:", "Update:" prefixes that add noise
    text = re.sub(r"^(Edit|Update|EDIT|UPDATE)\s*:?\s*", "", text)
 
    return text.strip()
 
 
def is_valid_pair(error: str, fix: str) -> bool:
    """Check if an error→fix pair meets quality thresholds."""
    if len(error) < MIN_ERROR_LENGTH:
        return False
    if len(fix) < MIN_FIX_LENGTH:
        return False
 
    # Fix should actually contain useful content
    if fix.lower().strip() in ["i have the same issue", "same here", "+1", "bump"]:
        return False
 
    # Error should contain something error-like
    error_indicators = [
        "error", "exception", "traceback", "failed", "fault",
        "warning", "cuda", "oom", "killed", "assert", "invalid",
    ]
    if not any(kw in error.lower() for kw in error_indicators):
        return False
 
    return True

def format_for_training(error: str, fix: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUCTION}\n\n```\n{error}\n```"},
            {"role": "assistant", "content": fix},
        ],
        # Keep metadata for analysis (stripped before training)
        "_meta": {
            "error_length": len(error),
            "fix_length": len(fix),
        },
    }

def load_raw_pairs() -> list[dict]:
    """Load all raw JSONL files from data/raw/."""
    pairs = []
    for jsonl_file in RAW_DIR.glob("*.jsonl"):
        print(f"  Loading {jsonl_file.name}...")
        with open(jsonl_file) as f:
            for line in f:
                if line.strip():
                    pairs.append(json.loads(line))
    return pairs

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    # Load raw data
    print("[*] Loading raw data...")
    raw_pairs = load_raw_pairs()
    print(f"    Loaded {len(raw_pairs)} raw pairs")
 
    if not raw_pairs:
        print("[!] No raw data found. Run scrapers first:")
        print("    python -m src.data.scrape_stackoverflow")
        print("    python -m src.data.scrape_github_issues")
        return
 
    # Clean and filter
    print("[*] Cleaning and filtering...")
    clean_pairs = []
    rejected = {"short_error": 0, "short_fix": 0, "no_error_signal": 0, "junk_fix": 0}
 
    for raw in raw_pairs:
        error = clean_error(raw.get("error", ""))
        fix = clean_fix(raw.get("fix", ""))
 
        if is_valid_pair(error, fix):
            formatted = format_for_training(error, fix)
            formatted["_meta"]["source_url"] = raw.get("source_url", "")
            formatted["_meta"]["source"] = (
                "stackoverflow" if "stackoverflow" in raw.get("source_url", "")
                else "github"
            )
            clean_pairs.append(formatted)
        else:
            if len(error) < MIN_ERROR_LENGTH:
                rejected["short_error"] += 1
            elif len(fix) < MIN_FIX_LENGTH:
                rejected["short_fix"] += 1
            else:
                rejected["no_error_signal"] += 1
 
    print(f"    Kept {len(clean_pairs)} valid pairs")
    print(f"    Rejected: {rejected}")
 
    # Deduplicate by error text (keep highest-scored)
    seen_errors = {}
    for p in clean_pairs:
        error_key = p["messages"][1]["content"][:200]  # first 200 chars as key
        if error_key not in seen_errors:
            seen_errors[error_key] = p
    clean_pairs = list(seen_errors.values())
    print(f"    After dedup: {len(clean_pairs)} pairs")
 
    # Shuffle and split (90/10 train/val)
    random.seed(42)
    random.shuffle(clean_pairs)
    split_idx = int(len(clean_pairs) * 0.9)
    train = clean_pairs[:split_idx]
    val = clean_pairs[split_idx:]
 
    for name, data in [("train", train), ("val", val)]:
        path = PROCESSED_DIR / f"{name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        print(f"    Wrote {len(data)} pairs to {path}")
 
    stats = {
        "total_raw": len(raw_pairs),
        "total_clean": len(clean_pairs),
        "train_size": len(train),
        "val_size": len(val),
        "rejected": rejected,
        "avg_error_length": sum(p["_meta"]["error_length"] for p in clean_pairs) / max(len(clean_pairs), 1),
        "avg_fix_length": sum(p["_meta"]["fix_length"] for p in clean_pairs) / max(len(clean_pairs), 1),
    }
    with open(PROCESSED_DIR / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
 
    print(f"\n[✓] Dataset ready!")
    print(f"    Train: {len(train)} | Val: {len(val)}")
    print(f"    Avg error length: {stats['avg_error_length']:.0f} chars")
    print(f"    Avg fix length: {stats['avg_fix_length']:.0f} chars")
 
 
if __name__ == "__main__":
    main()