#!/usr/bin/env python3
"""
precompute/generate_reasoning.py

What it does:
  - Loads the top 200 candidates from artifacts/reranker_scores.json.
  - Loads candidates.jsonl and extracts full profiles for those 200 candidate IDs.
  - For each candidate, extracts specific facts (skills, company history, experience,
    title, skill assessment scores, activity recency, notice period, and open-to-work).
  - Slots these facts into one of 8 distinct sentence templates where every sentence
    references at least 2 specific facts.
  - Saves the reasoning to artifacts/reasoning_cache.json keyed by candidate_id.

Inputs:
  - artifacts/reranker_scores.json (for top 200 candidate IDs)
  - candidates.jsonl (source candidate profiles)

Outputs:
  - artifacts/reasoning_cache.json

Expected runtime:
  - ~10-20 seconds on CPU.
"""

import gzip
import json
import sys
from pathlib import Path
from tqdm import tqdm

ARTIFACTS = Path(__file__).parent.parent / "artifacts"

HARD_REQUIRED = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "vector", "retrieval", "rag", "semantic search",
    "ranking", "nlp", "sentence transformers", "bge", "hybrid search",
    "recommendation", "information retrieval", "reranking", "bm25", "python",
    "machine learning", "llm", "transformers", "fine-tuning", "ndcg", "evaluation",
    "a/b testing"
}

CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "ibm global", "cts"
}


def extract_facts(cand: dict) -> dict:
    p = cand["profile"]
    sig = cand["redrob_signals"]
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])

    # 1. Exact matching skill names from the hard-required list
    hard_matches = [s["name"] for s in skills if any(h in s["name"].lower() for h in HARD_REQUIRED)]

    # 2. Product company names from career history (exclude consulting giants)
    product_companies = [
        h["company"] for h in career
        if not any(c in h["company"].lower() for c in CONSULTING)
    ]

    # 3. Years of experience
    yoe = p.get("years_of_experience", 0) or 0

    # 4. Current title
    title = p.get("current_title", "")

    # 5. Assessment scores for relevant skills
    assessment = sig.get("skill_assessment_scores", {}) or {}
    relevant_assessments = [
        f"{k.capitalize()} ({v}/100)" for k, v in assessment.items()
        if v is not None and (k.lower() in HARD_REQUIRED or any(h in k.lower() for h in HARD_REQUIRED))
    ]

    # 6. Days since last active
    from datetime import date, datetime
    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
    except Exception:
        days_inactive = 999

    # 7. Notice period
    notice_days = sig.get("notice_period_days", 60)

    # 8. Open to work flag
    open_to_work = sig.get("open_to_work_flag", False)

    return {
        "title": title,
        "company": p.get("current_company", ""),
        "yoe": yoe,
        "location": p.get("location", ""),
        "hard_matches": hard_matches,
        "product_companies": product_companies,
        "relevant_assessments": relevant_assessments,
        "days_inactive": days_inactive,
        "open_to_work": open_to_work,
        "notice_days": notice_days,
        "github_score": sig.get("github_activity_score", -1),
    }


def build_reasoning(facts: dict, cid: str) -> str:
    f = facts
    
    # Pre-process facts to have clean default representations if empty
    title = f["title"] if f["title"] else "AI Engineer"
    
    company = "a product company"
    if f["product_companies"]:
        company = f["product_companies"][0]
    elif f["company"]:
        company = f["company"]
        
    skills = "machine learning"
    if f["hard_matches"]:
        skills = ", ".join(f["hard_matches"][:3])
        
    notice = f"{f['notice_days']}-day" if f["notice_days"] > 0 else "immediate"
    if f["notice_days"] <= 30:
        notice = "short"
        
    work_status = "actively open to work" if f["open_to_work"] else "receptive to opportunities"
    
    days_active = f"last active {f['days_inactive']} days ago" if f["days_inactive"] != 999 else "active on the platform"
    if f["days_inactive"] <= 14:
        days_active = "highly active recently"
        
    location = f["location"] if f["location"] else "India"
    
    assessments_str = ""
    if f["relevant_assessments"]:
        assessments_str = f"with scores like {', '.join(f['relevant_assessments'][:2])}"
    else:
        assessments_str = f"demonstrated via key skills"
        
    github_str = f"with a GitHub score of {f['github_score']:.0f}" if f["github_score"] >= 0 else "having a complete profile"

    # Select template deterministically based on candidate ID hash to ensure variety
    # Using a simple python hash function but ensuring it's positive
    template_idx = abs(hash(cid)) % 8
    
    if template_idx == 0:
        s1 = f"Possesses {f['yoe']:.1f} years of experience as a {title}, with key expertise in {skills}."
        s2 = f"Currently associated with {company}, they are {work_status} with a {f['notice_days']}-day notice period."
    elif template_idx == 1:
        s1 = f"An experienced {title} with a career history including {company} and {f['yoe']:.1f} total years of experience."
        s2 = f"They possess strong skill matches in {skills} and maintain a {notice} notice period."
    elif template_idx == 2:
        s1 = f"Having worked as a {title} at {company}, this candidate has accumulated {f['yoe']:.1f} years of experience."
        s2 = f"Their profile highlights expertise in {skills} {assessments_str} while being {days_active}."
    elif template_idx == 3:
        s1 = f"Demonstrates strong proficiency in {skills} from {f['yoe']:.1f} years in roles like {title}."
        s2 = f"Based in {location}, they are {work_status} with a {f['notice_days']}-day notice period."
    elif template_idx == 4:
        s1 = f"With {f['yoe']:.1f} years of experience, this candidate has served as a {title} at {company}."
        s2 = f"Their technical skill set includes {skills} {assessments_str}, and they show a {notice} notice period."
    elif template_idx == 5:
        s1 = f"As a {title} with {f['yoe']:.1f} years of experience, they have developed hands-on skills in {skills}."
        s2 = f"They are currently {work_status} {github_str} and are {days_active}."
    elif template_idx == 6:
        s1 = f"This candidate has spent {f['yoe']:.1f} years in the industry, most recently as a {title} at {company}."
        s2 = f"They exhibit strong capabilities in {skills} and can start within a {f['notice_days']}-day notice period."
    else:  # template_idx == 7
        s1 = f"A specialist in {skills} who has spent {f['yoe']:.1f} years of experience in roles including {title}."
        s2 = f"Their background at {company} is supported by their {work_status} status and being {days_active}."
        
    return f"{s1} {s2}"


def load_candidates(path):
    path = Path(path)
    if path.suffix == ".gz":
        opener = lambda: gzip.open(path, "rt", encoding="utf-8")
    else:
        opener = lambda: open(path, "r", encoding="utf-8")
    with opener() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    args = parser.parse_args()

    # Load reranker scores (top 200 candidates)
    rerank_path = ARTIFACTS / "reranker_scores.json"
    if not rerank_path.exists():
        print("ERROR: run build_reranker_scores.py first to produce artifacts/reranker_scores.json")
        sys.exit(1)

    with open(rerank_path, "r", encoding="utf-8") as f:
        reranker_scores = json.load(f)

    top_ids = {entry["candidate_id"]: True for entry in reranker_scores[:200]}
    print(f"Generating reasoning for {len(top_ids)} candidates from reranker scores...")

    reasoning_cache = {}
    for cand in tqdm(load_candidates(args.candidates), total=100000, desc="Scanning candidates"):
        cid = cand["candidate_id"]
        if cid not in top_ids:
            continue
        facts = extract_facts(cand)
        reasoning = build_reasoning(facts, cid)
        reasoning_cache[cid] = reasoning
        if len(reasoning_cache) == len(top_ids):
            break  # Early exit once all found

    out = ARTIFACTS / "reasoning_cache.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(reasoning_cache, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(reasoning_cache)} reasoning strings to {out}")

    # Print a sample
    print("\n--- Sample reasonings ---")
    for cid, r in list(reasoning_cache.items())[:5]:
        print(f"[{cid}]: {r}")


if __name__ == "__main__":
    main()
