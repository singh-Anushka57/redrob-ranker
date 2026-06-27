#!/usr/bin/env python3
"""
precompute/build_features.py
Parses all 100K candidates into a flat feature parquet.
Run once. Produces artifacts/features.parquet
"""
import gzip, json, re
from datetime import date, datetime
from pathlib import Path
import pandas as pd
from tqdm import tqdm

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

CONSULTING = {
    "tcs", "tata consultancy", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware", "mindtree",
    "ltimindtree", "l&t infotech", "ibm global services", "cts",
}

BAD_TITLES = {
    "hr manager", "operations manager", "marketing manager", "content writer",
    "accountant", "civil engineer", "mechanical engineer", "customer support",
    "business analyst", "project manager", "sales manager", "product manager",
    "qa engineer", "financial analyst", "supply chain",
}

GOOD_TITLE_KW = {
    "machine learning", "ml engineer", "ai engineer", "nlp", "search engineer",
    "data scientist", "applied scientist", "research engineer", "recommendation",
    "ranking engineer", "retrieval", "software engineer", "backend engineer",
    "principal engineer", "senior engineer", "staff engineer",
}

PREFERRED_LOCS = {
    "noida", "pune", "hyderabad", "mumbai", "delhi", "gurgaon", "gurugram",
    "bangalore", "bengaluru", "ncr",
}


def parse_date(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except Exception:
        return None


def days_since(d):
    if d is None:
        return 999
    return (date.today() - d).days


def is_consulting(company_name: str) -> bool:
    cn = company_name.lower()
    return any(c in cn for c in CONSULTING)


def company_type(career_history):
    """Returns: 'product', 'mixed', 'consulting_only'"""
    companies = [h["company"] for h in career_history]
    consulting_flags = [is_consulting(c) for c in companies]
    if all(consulting_flags):
        return "consulting_only"
    if any(consulting_flags):
        return "mixed"
    return "product"


def title_score(title: str) -> float:
    tl = title.lower()
    if any(bad in tl for bad in BAD_TITLES):
        return 0.0
    if any(good in tl for good in GOOD_TITLE_KW):
        return 1.0
    # Partial engineering terms
    if any(kw in tl for kw in ["engineer", "developer", "scientist", "architect"]):
        return 0.6
    return 0.2


def location_score(location: str, country: str, willing_to_relocate: bool) -> float:
    loc_lower = (location or "").lower()
    if any(pref in loc_lower for pref in PREFERRED_LOCS):
        return 1.0
    if country and country.lower() == "india":
        if willing_to_relocate:
            return 0.85
        return 0.6
    # Outside India
    if willing_to_relocate:
        return 0.5
    return 0.2


def skill_text(skills) -> str:
    """Concatenate skill names for embedding."""
    return " ".join(s["name"] for s in skills)


def build_profile_text(cand) -> str:
    """Rich text for embedding: title + skills + career descriptions."""
    p = cand["profile"]
    parts = [
        p.get("headline", ""),
        p.get("summary", "")[:400],
    ]
    parts.append(skill_text(cand.get("skills", [])))
    for h in cand.get("career_history", [])[:3]:
        parts.append(h.get("title", ""))
        parts.append(h.get("description", "")[:300])
    return " ".join(filter(None, parts))


def featurize(cand) -> dict:
    p = cand["profile"]
    sig = cand["redrob_signals"]
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])

    last_active = parse_date(sig.get("last_active_date"))
    last_active_days = days_since(last_active)

    ctype = company_type(career)

    # Years of experience — use profile field, cross-check with career
    yoe = p.get("years_of_experience", 0) or 0

    # Skill names (lowercase) + assessment scores
    skill_names = [s["name"].lower() for s in skills]
    assessment = sig.get("skill_assessment_scores", {}) or {}

    # Location
    loc_sc = location_score(
        p.get("location", ""),
        p.get("country", ""),
        sig.get("willing_to_relocate", False),
    )

    # Title score
    t_sc = title_score(p.get("current_title", ""))

    # Availability multiplier components (stored raw, fused in ranker)
    rrr = sig.get("recruiter_response_rate", 0.5)
    notice = sig.get("notice_period_days", 60)
    open_work = sig.get("open_to_work_flag", False)
    icr = sig.get("interview_completion_rate", 0.5)
    oar = sig.get("offer_acceptance_rate", -1)

    # Honeypot detection signals
    # 1) Expert in too many assessment skills with impossible scores
    expert_count = sum(1 for s in skills if s.get("proficiency") == "expert")
    assessment_expert_count = sum(1 for v in assessment.values() if v >= 88)

    # 2) tenure inconsistency: sum of durations > stated yoe * 12 + 24
    total_tenure_months = sum(h.get("duration_months", 0) for h in career)
    tenure_inflation = total_tenure_months > (yoe * 12 + 30)

    return {
        "candidate_id": cand["candidate_id"],
        "current_title": p.get("current_title", ""),
        "location": p.get("location", ""),
        "country": p.get("country", ""),
        "years_of_experience": yoe,
        "company_type": ctype,
        "skill_names_csv": ",".join(skill_names),
        "skill_count": len(skills),
        "expert_skill_count": expert_count,
        "assessment_expert_count": assessment_expert_count,
        "tenure_inflation": int(tenure_inflation),
        "title_score": t_sc,
        "location_score": loc_sc,
        # Raw behavioral signals
        "last_active_days": last_active_days,
        "open_to_work": int(open_work),
        "recruiter_response_rate": rrr,
        "avg_response_time_hours": sig.get("avg_response_time_hours", 24),
        "notice_period_days": notice,
        "github_activity_score": sig.get("github_activity_score", -1),
        "interview_completion_rate": icr,
        "offer_acceptance_rate": oar,
        "profile_completeness_score": sig.get("profile_completeness_score", 0),
        "saved_by_recruiters_30d": sig.get("saved_by_recruiters_30d", 0),
        "applications_submitted_30d": sig.get("applications_submitted_30d", 0),
        "preferred_work_mode": sig.get("preferred_work_mode", ""),
        "willing_to_relocate": int(sig.get("willing_to_relocate", False)),
        "verified_email": int(sig.get("verified_email", False)),
        "verified_phone": int(sig.get("verified_phone", False)),
        "profile_text": build_profile_text(cand),
    }


def load_candidates(path: Path):
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
    import sys
    candidates_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("candidates.jsonl")
    if not candidates_path.exists():
        # Try gz version
        gz = candidates_path.with_suffix(".jsonl.gz")
        if gz.exists():
            candidates_path = gz
        else:
            print(f"ERROR: cannot find {candidates_path}")
            sys.exit(1)

    print(f"Loading candidates from {candidates_path} ...")
    rows = []
    for cand in tqdm(load_candidates(candidates_path), total=100000, desc="Featurizing"):
        rows.append(featurize(cand))

    df = pd.DataFrame(rows)
    out = ARTIFACTS / "features.parquet"
    df.to_parquet(out, index=False)
    print(f"\nSaved {len(df)} rows to {out}")
    print(df.dtypes)
    print("\nCompany type distribution:")
    print(df["company_type"].value_counts())


if __name__ == "__main__":
    main()
