#!/usr/bin/env python3
"""
precompute/build_behavioral_scores.py

What it does:
  - Loads the top 500 candidates from artifacts/reranker_scores.json.
  - Loads features for all candidates from artifacts/features.parquet.
  - Calculates a hire probability score (0 to 1) for each of these 500 candidates
    based on behavioral signals (responsiveness, interview intent, offer acceptance rate,
    recency of activity, and notice period).
  - Saves the results as a JSON dictionary mapping candidate_id to hire_probability float
    in artifacts/behavioral_scores.json.

Inputs:
  - artifacts/reranker_scores.json
  - artifacts/features.parquet

Outputs:
  - artifacts/behavioral_scores.json

Expected runtime:
  - < 2 seconds on CPU.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent.parent / "artifacts"


def main():
    print("Loading data...")
    # Load reranker scores
    rerank_path = ARTIFACTS / "reranker_scores.json"
    if not rerank_path.exists():
        print("ERROR: run build_reranker_scores.py first")
        import sys; sys.exit(1)
        
    with open(rerank_path, "r", encoding="utf-8") as f:
        reranker_scores = json.load(f)
        
    top_500_ids = [entry["candidate_id"] for entry in reranker_scores]

    # Load features
    feat_path = ARTIFACTS / "features.parquet"
    if not feat_path.exists():
        print("ERROR: run build_features.py first")
        import sys; sys.exit(1)
        
    df = pd.read_parquet(feat_path)
    # Set index to candidate_id for fast lookup
    df_filtered = df[df["candidate_id"].isin(top_500_ids)].set_index("candidate_id")

    behavioral_scores = {}
    for cid in top_500_ids:
        if cid not in df_filtered.index:
            # Fallback default value if missing
            behavioral_scores[cid] = 0.5
            continue
            
        row = df_filtered.loc[cid]
        
        # 1. Responsiveness
        rrr = float(row["recruiter_response_rate"])
        resp_time = float(row["avg_response_time_hours"])
        # responsiveness = recruiter_response_rate * (1 / log1p(avg_response_time_hours + 1))
        responsiveness = rrr * (1.0 / np.log1p(resp_time + 1.0))
        
        # 2. Interview Intent
        icr = float(row["interview_completion_rate"])
        open_to_work = int(row["open_to_work"])
        open_to_work_bonus = 1.2 if open_to_work == 1 else 1.0
        interview_intent = icr * open_to_work_bonus
        
        # 3. Offer Intent
        oar = float(row["offer_acceptance_rate"])
        offer_intent = oar if oar >= 0.0 else 0.5
        
        # 4. Recency Factor
        days = float(row["last_active_days"])
        # exponential decay (half-life = 45 days)
        recency_factor = np.exp(-days * np.log(2.0) / 45.0)
        
        # 5. Notice Factor
        notice = float(row["notice_period_days"])
        if notice <= 30:
            notice_factor = 1.0
        elif notice <= 60:
            notice_factor = 0.85
        elif notice <= 90:
            notice_factor = 0.70
        else:
            notice_factor = 0.55
            
        # Composite score
        hire_prob = (
            responsiveness * 0.30 +
            interview_intent * 0.30 +
            offer_intent * 0.20 +
            recency_factor * 0.20
        ) * notice_factor
        
        # Clip to [0, 1]
        hire_prob_clipped = float(np.clip(hire_prob, 0.0, 1.0))
        behavioral_scores[cid] = hire_prob_clipped

    # Save behavioral scores
    out_path = ARTIFACTS / "behavioral_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(behavioral_scores, f, indent=2, ensure_ascii=False)
        
    print(f"Saved behavioral scores to {out_path} for {len(behavioral_scores)} candidates")


if __name__ == "__main__":
    main()
