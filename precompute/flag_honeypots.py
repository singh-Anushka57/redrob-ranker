#!/usr/bin/env python3
"""
precompute/flag_honeypots.py
Detects candidates with impossible/inconsistent profiles.
Produces artifacts/honeypot_flags.json
"""
import gzip, json, sys
from pathlib import Path
from datetime import date, datetime
from tqdm import tqdm

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)


def parse_year(s):
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m-%d").year
    except Exception:
        return None


KNOWN_FOUNDING_YEARS = {
    # A few well-known companies to catch obvious traps
    "zomato": 2010, "swiggy": 2014, "meesho": 2015, "cred": 2018,
    "razorpay": 2014, "zepto": 2021, "blinkit": 2013, "dunzo": 2014,
    "groww": 2016, "zerodha": 2010, "ola": 2010, "rapido": 2015,
    "mad street den": 2013, "sarvam ai": 2023, "krutrim": 2023,
    "sarvam": 2023,
}


def flag_candidate(cand) -> list[str]:
    flags = []
    p = cand["profile"]
    sig = cand["redrob_signals"]
    career = cand.get("career_history", [])
    skills = cand.get("skills", [])

    yoe = p.get("years_of_experience", 0) or 0

    # --- FLAG 1: Expert breadth impossibility ---
    # Claiming expert in 8+ distinct domain categories is suspicious
    expert_skills = [s for s in skills if s.get("proficiency") == "expert"]
    if len(expert_skills) >= 8:
        flags.append(f"expert_breadth:{len(expert_skills)}")

    # Assessment scores: 5+ assessments all >= 88 is statistically implausible
    assessment = sig.get("skill_assessment_scores", {}) or {}
    high_assessments = [k for k, v in assessment.items() if v >= 88]
    if len(high_assessments) >= 5:
        flags.append(f"impossible_assessment_breadth:{len(high_assessments)}")

    # --- FLAG 2: Tenure inflation ---
    total_tenure_months = sum(h.get("duration_months", 0) for h in career)
    # Allow 12 months slack (overlapping roles, breaks)
    if total_tenure_months > (yoe * 12 + 36):
        flags.append(f"tenure_inflation:total={total_tenure_months}mo,yoe={yoe}yr")

    # --- FLAG 3: Company founding year inconsistency ---
    for h in career:
        company_lower = h.get("company", "").lower()
        start_year = parse_year(h.get("start_date"))
        for known_co, founding_year in KNOWN_FOUNDING_YEARS.items():
            if known_co in company_lower and start_year and start_year < founding_year:
                flags.append(f"impossible_tenure:{h['company']}_start={start_year}_founded={founding_year}")

    # --- FLAG 4: Profile completeness 100 but never active (zombie profile) ---
    completeness = sig.get("profile_completeness_score", 0)
    last_active_str = sig.get("last_active_date", "")
    signup_str = sig.get("signup_date", "")
    if completeness >= 99 and last_active_str and signup_str:
        try:
            last_active = datetime.strptime(last_active_str, "%Y-%m-%d").date()
            signup = datetime.strptime(signup_str, "%Y-%m-%d").date()
            days_diff = (last_active - signup).days
            if days_diff <= 1:
                flags.append("zombie_profile:completeness=100_never_returned")
        except Exception:
            pass

    # --- FLAG 5: Years of experience impossible given age proxy ---
    # If someone has 25+ years experience but education ended very recently
    edu = cand.get("education", [])
    if edu and yoe > 20:
        max_end_year = max((e.get("end_year", 0) or 0) for e in edu)
        if max_end_year >= 2015:  # Graduated after 2015 but claims 20+ yrs
            flags.append(f"age_impossibility:yoe={yoe}_grad_year={max_end_year}")

    return flags


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
    candidates_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("candidates.jsonl")
    if not candidates_path.exists():
        gz = candidates_path.with_suffix(".jsonl.gz")
        if gz.exists():
            candidates_path = gz
        else:
            print(f"ERROR: cannot find {candidates_path}")
            sys.exit(1)

    flagged = {}
    total = 0
    for cand in tqdm(load_candidates(candidates_path), total=100000, desc="Honeypot scan"):
        total += 1
        flags = flag_candidate(cand)
        if flags:
            flagged[cand["candidate_id"]] = flags

    out = ARTIFACTS / "honeypot_flags.json"
    with open(out, "w") as f:
        json.dump(flagged, f, indent=2)

    print(f"\nScanned {total} candidates. Flagged {len(flagged)} potential honeypots.")
    print(f"Saved to {out}")

    # Show flag distribution
    all_flag_types = {}
    for flags in flagged.values():
        for flag in flags:
            key = flag.split(":")[0]
            all_flag_types[key] = all_flag_types.get(key, 0) + 1
    print("\nFlag type distribution:")
    for k, v in sorted(all_flag_types.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
