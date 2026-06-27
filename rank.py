#!/usr/bin/env python3
"""
rank.py — Main entry point for the Redrob candidate ranker.

What it does:
  - Loads precomputed offline artifacts (cross-encoder/BGE reranker scores,
    behavioral hire probability scores, features, honeypot flags, and reasoning cache).
  - Normalizes semantic BGE and Cross-Encoder scores.
  - Computes composite fit scores and applies behavioral conversions and penalties.
  - Generates the top 100 candidates with non-increasing scores.
  - Formats output as a CSV compliant with validate_submission.py.

Usage:
  python rank.py --candidates ./candidates.jsonl --out ./submission.csv

Inputs:
  - --candidates (defaults to candidates.jsonl)
  - artifacts/features.parquet
  - artifacts/honeypot_flags.json
  - artifacts/reranker_scores.json
  - artifacts/behavioral_scores.json
  - artifacts/reasoning_cache.json

Outputs:
  - --out (defaults to submission.csv)

Constraints:
  - Runtime <= 5 minutes (CPU-only, no network).
"""

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent / "artifacts"

# ----- Constants and Rules from pipeline.py -----
HARD_REQUIRED_SKILLS = [
    'embeddings','faiss','pinecone','weaviate','qdrant','milvus',
    'opensearch','elasticsearch','vector','retrieval','rag',
    'semantic search','hybrid search','ranking','reranking',
    'nlp','sentence transformers','bge','recommendation',
    'information retrieval','bm25','python','machine learning',
    'llm','transformers','fine-tuning','evaluation','ndcg',
    'haystack','mlflow','huggingface','pytorch','tensorflow',
]

RETRIEVAL_SKILLS = {
    'embedding','faiss','vector','retrieval','rag','semantic','hybrid','ranking',
    'reranking','recommendation','bm25','opensearch','elasticsearch','pinecone',
    'weaviate','qdrant','haystack','nlp','information retrieval','fine-tuning llms',
    'fine-tuning','sentence transformers','llm','transformers','machine learning',
}

CONSULTING = {
    'tcs','tata consultancy','infosys','wipro','accenture','cognizant',
    'capgemini','hcl','tech mahindra','mphasis','hexaware','mindtree',
    'ltimindtree','ibm global','cts','l&t infotech'
}

BAD_TITLES = {
    'hr manager','operations manager','marketing manager','content writer',
    'accountant','civil engineer','mechanical engineer','customer support',
    'business analyst','project manager','sales manager','qa engineer',
    'financial analyst','supply chain','graphic designer'
}

HARD_REQUIRED_GEN = {
    "embeddings", "faiss", "pinecone", "weaviate", "qdrant", "opensearch",
    "elasticsearch", "vector", "retrieval", "rag", "semantic search",
    "ranking", "nlp", "sentence transformers", "bge", "hybrid search",
    "recommendation", "information retrieval", "reranking", "bm25", "python",
    "machine learning", "llm", "transformers", "fine-tuning", "ndcg", "evaluation",
    "a/b testing"
}

# =====================================================================
# SCORING HELPERS
# =====================================================================

def skill_score(skills) -> float:
    if not skills:
        return 0.0
    names = ' '.join(s.get('name', '').lower() for s in skills)
    matches = sum(1 for r in HARD_REQUIRED_SKILLS if r in names)
    advanced_bonus = sum(
        0.5 for s in skills
        if s.get('proficiency') in ('advanced', 'expert')
        and any(r in s.get('name', '').lower() for r in RETRIEVAL_SKILLS)
    )
    return min((matches + advanced_bonus) / 10.0, 1.0)


def exp_score(yoe: float, company_type: str) -> float:
    if yoe < 2:
        base = 0.1
    elif yoe < 4:
        base = 0.4
    elif yoe <= 9:
        base = 0.7 + (min(yoe, 8) - 4) / 4 * 0.3
    elif yoe <= 12:
        base = 0.9
    else:
        base = 0.7
    bonus = {'product': 0.15, 'mixed': 0.05, 'consulting_only': -0.30}
    return max(0.0, min(1.0, base + bonus.get(company_type, 0.0)))


def title_score(title: str) -> float:
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.0
    good = [
        'machine learning', 'ml engineer', 'ai engineer', 'nlp',
        'search engineer', 'data scientist', 'applied scientist',
        'research engineer', 'recommendation', 'retrieval',
        'ranking engineer', 'principal engineer', 'staff engineer',
    ]
    if any(g in tl for g in good):
        return 1.0
    if any(k in tl for k in ['engineer', 'developer', 'scientist', 'architect']):
        return 0.6
    return 0.2


def location_score(loc: str, country: str, rel: bool) -> float:
    preferred = {'noida','pune','hyderabad','mumbai','delhi','gurgaon','gurugram','bangalore','bengaluru','ncr'}
    if any(p in loc.lower() for p in preferred):
        return 1.0
    if country == 'India':
        return 0.85 if rel else 0.6
    return 0.5 if rel else 0.2


def disqualifier_penalty(title: str, company_type: str) -> float:
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.05
    if company_type == 'consulting_only':
        return 0.25
    return 1.0


# =====================================================================
# FALLBACK REASONING GENERATION (FACT-ASSEMBLY)
# =====================================================================

def extract_facts(cand: dict) -> dict:
    p = cand.get("profile", {})
    sig = cand.get("redrob_signals", {})
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])

    hard_matches = [s["name"] for s in skills if any(h in s["name"].lower() for h in HARD_REQUIRED_GEN)]
    product_companies = [
        h["company"] for h in career
        if not any(c in h["company"].lower() for c in CONSULTING)
    ]
    yoe = p.get("years_of_experience", 0) or 0
    title = p.get("current_title", "")
    
    assessment = sig.get("skill_assessment_scores", {}) or {}
    relevant_assessments = [
        f"{k.capitalize()} ({v}/100)" for k, v in assessment.items()
        if v is not None and (k.lower() in HARD_REQUIRED_GEN or any(h in k.lower() for h in HARD_REQUIRED_GEN))
    ]

    from datetime import date, datetime
    last_active_str = sig.get("last_active_date", "")
    try:
        last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
        days_inactive = (date.today() - last_active).days
    except Exception:
        days_inactive = 999

    notice_days = sig.get("notice_period_days", 60)
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


def build_fallback_reasoning(facts: dict, cid: str) -> str:
    f = facts
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


# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Timed Fast Ranker")
    parser.add_argument("--candidates", default="candidates.jsonl", help="Path to candidates.jsonl")
    parser.add_argument("--out", default="submission.csv", help="Output CSV file path")
    args = parser.parse_args()

    t0 = time.time()

    print("[1/4] Loading precomputed artifacts...")
    # Load reranker scores
    rerank_path = ARTIFACTS / "reranker_scores.json"
    if not rerank_path.exists():
        print("ERROR: reranker_scores.json not found in artifacts/.")
        sys.exit(1)
    with open(rerank_path, "r", encoding="utf-8") as f:
        reranker_data = json.load(f)  # list of {candidate_id, bge_score, crossencoder_score}
        
    top_500_ids = [entry["candidate_id"] for entry in reranker_data]
    top_500_set = set(top_500_ids)

    # Load behavioral scores
    behavioral_path = ARTIFACTS / "behavioral_scores.json"
    if not behavioral_path.exists():
        print("ERROR: behavioral_scores.json not found in artifacts/.")
        sys.exit(1)
    with open(behavioral_path, "r", encoding="utf-8") as f:
        behavioral_scores = json.load(f)  # dict candidate_id -> hire_probability

    # Load reasoning cache
    reasoning_path = ARTIFACTS / "reasoning_cache.json"
    reasoning_cache = {}
    if reasoning_path.exists():
        with open(reasoning_path, "r", encoding="utf-8") as f:
            reasoning_cache = json.load(f)

    # Load honeypot flags
    honeypot_path = ARTIFACTS / "honeypot_flags.json"
    honeypot_flags = {}
    if honeypot_path.exists():
        with open(honeypot_path, "r", encoding="utf-8") as f:
            honeypot_flags = json.load(f)

    # Load features
    feat_path = ARTIFACTS / "features.parquet"
    if not feat_path.exists():
        print("ERROR: features.parquet not found in artifacts/.")
        sys.exit(1)
    df = pd.read_parquet(feat_path)
    df_filtered = df[df["candidate_id"].isin(top_500_ids)].set_index("candidate_id")

    # Load raw candidates (for accurate skill_score matching)
    print("[2/4] Reading candidates JSONL for raw profiles...")
    cand_profiles = {}
    with open(args.candidates, "r", encoding="utf-8") as f:
        for line in f:
            # Fast candidate_id extraction to avoid json parsing unless matching
            try:
                cid = line.split('"', 4)[3]
                if cid in top_500_set:
                    cand_profiles[cid] = json.loads(line)
            except Exception:
                # Fallback to standard json loads if split fails
                try:
                    cand = json.loads(line)
                    cid = cand["candidate_id"]
                    if cid in top_500_set:
                        cand_profiles[cid] = cand
                except Exception:
                    pass

    # Normalize reranker scores
    bge_scores = np.array([entry["bge_score"] for entry in reranker_data])
    cross_scores = np.array([entry["crossencoder_score"] for entry in reranker_data])
    
    bge_min, bge_max = bge_scores.min(), bge_scores.max()
    cross_min, cross_max = cross_scores.min(), cross_scores.max()
    
    bge_norm_map = {}
    cross_norm_map = {}
    for entry in reranker_data:
        cid = entry["candidate_id"]
        bge_norm_map[cid] = (entry["bge_score"] - bge_min) / (bge_max - bge_min + 1e-9)
        cross_norm_map[cid] = (entry["crossencoder_score"] - cross_min) / (cross_max - cross_min + 1e-9)

    print("[3/4] Scoring candidates...")
    scored_candidates = []
    for cid in top_500_ids:
        if cid not in df_filtered.index:
            continue
            
        row = df_filtered.loc[cid]
        cand = cand_profiles.get(cid, {})
        
        # 1. Semantic Score
        bge_norm = bge_norm_map.get(cid, 0.0)
        cross_norm = cross_norm_map.get(cid, 0.0)
        semantic = 0.65 * cross_norm + 0.35 * bge_norm
        
        # 2. Sub-scores
        sk_score = skill_score(cand.get("skills", []))
        ex_score = exp_score(float(row["years_of_experience"]), str(row["company_type"]))
        t_score = title_score(str(row["current_title"]))
        l_score = location_score(str(row["location"]), str(row["country"]), bool(row["willing_to_relocate"]))
        
        # 3. Fit Score
        fit_score = (
            0.50 * semantic +
            0.20 * sk_score +
            0.15 * ex_score +
            0.08 * t_score +
            0.07 * l_score
        )
        
        # 4. Hire Intent Multiplier
        hire_prob = behavioral_scores.get(cid, 0.5)
        final_score = fit_score * (0.75 + 0.25 * hire_prob)
        
        # 5. Penalties
        disq = disqualifier_penalty(str(row["current_title"]), str(row["company_type"]))
        honeypot = 0.01 if cid in honeypot_flags else 1.0
        
        final_score = final_score * disq * honeypot
        final_score = float(np.clip(final_score, 0.0, 1.0))
        
        scored_candidates.append({
            "candidate_id": cid,
            "final_score": final_score,
            "current_title": str(row["current_title"]),
            "location": str(row["location"]),
            "is_honeypot": cid in honeypot_flags,
            "cand_raw": cand
        })

    # Sort: final_score descending, candidate_id ascending for tie-break
    scored_candidates.sort(key=lambda x: (-x["final_score"], x["candidate_id"]))
    
    # Get top 100
    top_100 = scored_candidates[:100]

    # Validate honeypot rate
    honeypot_count = sum(1 for x in top_100 if x["is_honeypot"])
    print(f"Honeypots in top 100: {honeypot_count}/100 ({honeypot_count:.1%})")
    assert honeypot_count < 10, f"Honeypot rate is too high: {honeypot_count}%! Check scoring logic."

    print("[4/4] Writing submission CSV...")
    # Write output
    out_path = Path(args.out)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        prev_score = None
        for rank, item in enumerate(top_100, start=1):
            cid = item["candidate_id"]
            score = round(item["final_score"], 6)
            
            # Ensure non-increasing scores by rank
            if prev_score is not None and score > prev_score:
                score = prev_score
            prev_score = score
            
            # Get reasoning from cache, fallback to generating it
            reasoning = reasoning_cache.get(cid, "")
            if not reasoning:
                facts = extract_facts(item["cand_raw"])
                reasoning = build_fallback_reasoning(facts, cid)
                
            writer.writerow([cid, rank, score, reasoning])

    total_time = time.time() - t0
    print(f"\nDone! Submission saved to {out_path} in {total_time:.2f} seconds.")
    print("\nTop 10 Candidates:")
    for rank, item in enumerate(top_100[:10], start=1):
        print(f"  #{rank:2d} | {item['candidate_id']} | Score: {item['final_score']:.6f} | "
              f"Title: {item['current_title'][:30]:<30} | Loc: {item['location']}")


if __name__ == "__main__":
    main()
