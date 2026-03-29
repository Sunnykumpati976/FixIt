"""
Scrape Stack Overflow for PyTorch/YOLO error→fix pairs.

Uses the StackExchange API (no auth required, ~300 req/day quota).
Targets questions with accepted/upvoted answers across relevant tags.

Output: data/raw/stackoverflow_pairs.jsonl
"""
import json
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.config import (
    STACKOVERFLOW_API,
    STACKOVERFLOW_TAGS,
    MAX_QUESTIONS_PER_TAG,
    MIN_ANSWER_SCORE,
)

RAW_DIR = Path("data/raw")

def fetch_questions(tag: str, page: int = 1, pagesize: int = 50) -> dict:
    resp = requests.get(
        f"{STACKOVERFLOW_API}/questions",
        params={
            "order": "desc",
            "sort": "votes",
            "tagged" : tag,
            "filter": "withbody",
            "site": "stackoverflow",
            "page": page,
            "pagesize": pagesize,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def fetch_answers(question_ids: list[int]) -> dict:
    ids_str = ";".join(str(qid) for qid in question_ids)
    resp = requests.get(
        f"{STACKOVERFLOW_API}/questions/{ids_str}/answers",
        params={
            "order": "desc",
            "sort": "votes",
            "filter": "withbody",
            "site": "stackoverflow",
            "pagesize": 100,
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for code in soup.find_all("code"):
        code.string = "\n" + code.get_text() + "\n"
    return soup.get_text(separator="\n").strip()

def extract_error_from_question(body_text: str) -> str | None:
    traceback_pattern = re.compile(
        r"(Traceback \(most recent call last\):.+?)(?=\n\n|\n[A-Z]|\Z)",
        re.DOTALL,
    )
    match = traceback_pattern.search(body_text)
    if match:
        return match.group(1).strip()
 
    error_patterns = [
        r"((?:RuntimeError|ValueError|TypeError|AttributeError|"
        r"ModuleNotFoundError|ImportError|FileNotFoundError|"
        r"CUDA error|cuda runtime error|OOM|"
        r"AssertionError|IndexError|KeyError|"
        r"torch\.\w+Error).+?)(?=\n\n|\Z)",
    ]
    for pat in error_patterns:
        match = re.search(pat, body_text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
 
    # Fallback: if there's a code block that looks error-ish
    code_blocks = re.findall(r"```\n(.+?)\n```", body_text, re.DOTALL)
    for block in code_blocks:
        if any(kw in block.lower() for kw in ["error", "exception", "traceback", "failed"]):
            return block.strip()
 
    return None

def scrape_tag(tag: str) -> list[dict]:
    """
    Scrape all error→fix pairs for a single SO tag.
    Returns list of {error, fix, source_url, tag, score} dicts.
    """
    pairs = []
    pages_needed = (MAX_QUESTIONS_PER_TAG // 50) + 1
 
    for page in range(1, pages_needed + 1):
        try:
            data = fetch_questions(tag, page=page)
        except requests.RequestException as e:
            print(f"  [!] API error on page {page}: {e}")
            break
 
        questions = data.get("items", [])
        if not questions:
            break
 
        # Batch fetch answers
        q_ids = [q["question_id"] for q in questions]
        try:
            answers_data = fetch_answers(q_ids)
        except requests.RequestException as e:
            print(f"  [!] Answer fetch error: {e}")
            continue
 
        # Index answers by question_id
        answers_by_q: dict[int, list] = {}
        for ans in answers_data.get("items", []):
            qid = ans["question_id"]
            answers_by_q.setdefault(qid, []).append(ans)
 
        for q in questions:
            qid = q["question_id"]
            q_body = html_to_text(q.get("body", ""))
            error = extract_error_from_question(q_body)
 
            if not error:
                continue  # Skip questions where we can't find an error
 
            # Get best answer (accepted or highest score)
            q_answers = answers_by_q.get(qid, [])
            best_answer = None
            for ans in q_answers:
                if ans.get("is_accepted"):
                    best_answer = ans
                    break
            if not best_answer and q_answers:
                q_answers.sort(key=lambda a: a.get("score", 0), reverse=True)
                if q_answers[0].get("score", 0) >= MIN_ANSWER_SCORE:
                    best_answer = q_answers[0]
 
            if not best_answer:
                continue
 
            fix = html_to_text(best_answer.get("body", ""))
            if not fix:
                continue
 
            pairs.append({
                "error": error,
                "fix": fix,
                "source_url": q.get("link", ""),
                "tag": tag,
                "question_score": q.get("score", 0),
                "answer_score": best_answer.get("score", 0),
            })
 
        # Respect API rate limits
        quota_remaining = data.get("quota_remaining", 0)
        if quota_remaining < 10:
            print(f"  [!] API quota low ({quota_remaining}), stopping")
            break
        time.sleep(1)  # Be polite
 
    return pairs

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "stackoverflow_pairs.jsonl"
 
    all_pairs = []
    for tag in tqdm(STACKOVERFLOW_TAGS, desc="Scraping SO tags"):
        print(f"\n[*] Scraping tag: {tag}")
        pairs = scrape_tag(tag)
        print(f"    Found {len(pairs)} pairs")
        all_pairs.extend(pairs)
        time.sleep(2)  # Extra delay between tags
 
    # Deduplicate by source_url
    seen_urls = set()
    unique_pairs = []
    for p in all_pairs:
        if p["source_url"] not in seen_urls:
            seen_urls.add(p["source_url"])
            unique_pairs.append(p)
 
    # Write JSONL (one JSON object per line — standard ML data format)
    with open(output_path, "w") as f:
        for pair in unique_pairs:
            f.write(json.dumps(pair) + "\n")
 
    print(f"\n[✓] Saved {len(unique_pairs)} unique pairs to {output_path}")
    print(f"    Duplicates removed: {len(all_pairs) - len(unique_pairs)}")
 
 
if __name__ == "__main__":
    main()

