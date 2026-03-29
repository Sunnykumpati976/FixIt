import json
import os
import re
import time
from pathlib import Path
 
import requests
from tqdm import tqdm
 
from src.config import GITHUB_REPOS, MAX_ISSUES_PER_REPO
 
RAW_DIR = Path("data/raw")

def get_headers() -> dict:
    token = os.environ.get("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github.v3+json"}
    if token:
        headers["Authorization"] = f"token {token}"
    return headers

def fetch_issues(repo: str, page: int = 1, per_page: int = 50, labels: str | None = "bug") -> list[dict]:
    params = {
        "state": "closed",
        "sort": "comments",
        "direction": "desc",
        "page": page,
        "per_page": per_page,
    }
    if labels:
        params["labels"] = labels
    
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/issues",
        headers=get_headers(),
        params=params,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def fetch_comments(repo: str, issue_number: int) -> list[dict]:
    resp = requests.get(
        f"https://api.github.com/repos/{repo}/issues/{issue_number}/comments",
        headers=get_headers(),
        params={"per_page": 100},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()

def extract_error_from_issue(body: str) -> str | None:

    if not body:
        return None
 
    # Look for code blocks containing errors
    code_blocks = re.findall(r"```[\w]*\n(.+?)\n```", body, re.DOTALL)
    for block in code_blocks:
        if any(kw in block.lower() for kw in [
            "error", "exception", "traceback", "failed", "fault",
            "runtimeerror", "cuda", "assert", "killed", "oom",
        ]):
            return block.strip()
 
    # Look for inline tracebacks
    traceback_match = re.search(
        r"(Traceback \(most recent call last\):.+?)(?=\n\n|\Z)",
        body, re.DOTALL,
    )
    if traceback_match:
        return traceback_match.group(1).strip()
 
    # Look for error lines outside code blocks
    error_match = re.search(
        r"((?:RuntimeError|ValueError|TypeError|ModuleNotFoundError|"
        r"CUDA error|AssertionError|FileNotFoundError).+?)(?=\n\n|\Z)",
        body, re.DOTALL | re.IGNORECASE,
    )
    if error_match:
        return error_match.group(1).strip()
 
    return None

def extract_fix_from_comments(
comments: list[dict]
) -> tuple[str | None, str | None]:
    """
    Find the most likely "fix" comment.
 
    Heuristics (in priority order):
    1. Comment by a repo maintainer/collaborator with code
    2. Most upvoted comment with code
    3. Comment referencing a PR or commit
 
    Returns (fix_text, commenter_association).
    """
    # Score each comment
    scored = []
    for c in comments:
        text = c.get("body", "")
        if not text or len(text) < 30:
            continue
 
        score = 0
        assoc = c.get("author_association", "NONE")
 
        # Maintainer/member comments are gold
        if assoc in ("MEMBER", "COLLABORATOR", "OWNER"):
            score += 10
        elif assoc == "CONTRIBUTOR":
            score += 5
 
        # Has code blocks (likely a fix)
        if "```" in text:
            score += 5
 
        # References a fix/solution
        fix_keywords = ["fix", "solved", "solution", "workaround", "try this",
                        "you need to", "you should", "the issue is", "this is because"]
        if any(kw in text.lower() for kw in fix_keywords):
            score += 3
 
        # References a PR
        if re.search(r"#\d+|pull/\d+|commit", text):
            score += 2
 
        # Reactions (upvotes)
        reactions = c.get("reactions", {})
        score += reactions.get("+1", 0) * 2
        score += reactions.get("heart", 0)
 
        scored.append((score, text, assoc))
 
    if not scored:
        return None, None
 
    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_text, best_assoc = scored[0]
 
    if best_score < 3:  # Too low confidence
        return None, None
 
    return best_text, best_assoc

def scrape_repo(repo: str) -> list[dict]:
    pairs = []
    pages_needed = (MAX_ISSUES_PER_REPO // 50) + 1
    labels = None if "pytorch/pytorch" in repo else "bug"

    for page in range(1, pages_needed + 1):
        try:
            issues = fetch_issues(repo, page=page, labels=labels)
        except requests.RequestException as e:
            print(f"  [!] API error: {e}")
            break
 
        if not issues:
            break
 
        # Filter out pull requests (GitHub API returns them as issues too)
        issues = [i for i in issues if "pull_request" not in i]
 
        for issue in tqdm(issues, desc=f"  Page {page}", leave=False):
            body = issue.get("body", "")
            error = extract_error_from_issue(body)
            if not error:
                continue
 
            try:
                comments = fetch_comments(repo, issue["number"])
            except requests.RequestException:
                continue
 
            fix, assoc = extract_fix_from_comments(comments)
            if not fix:
                continue
 
            pairs.append({
                "error": error,
                "fix": fix,
                "source_url": issue.get("html_url", ""),
                "repo": repo,
                "commenter_type": assoc,
                "issue_number": issue["number"],
                "labels": [l["name"] for l in issue.get("labels", [])],
            })
 
            time.sleep(0.5)  # Rate limit kindness
 
        time.sleep(1)
 
    return pairs

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / "github_issues_pairs.jsonl"
 
    all_pairs = []
    for repo in GITHUB_REPOS:
        print(f"\n[*] Scraping repo: {repo}")
        pairs = scrape_repo(repo)
        print(f"    Found {len(pairs)} pairs")
        all_pairs.extend(pairs)
 
    # Deduplicate by source_url
    seen = set()
    unique = []
    for p in all_pairs:
        if p["source_url"] not in seen:
            seen.add(p["source_url"])
            unique.append(p)
 
    with open(output_path, "w") as f:
        for pair in unique:
            f.write(json.dumps(pair) + "\n")
 
    print(f"\n[✓] Saved {len(unique)} pairs to {output_path}")
 
 
if __name__ == "__main__":
    main()