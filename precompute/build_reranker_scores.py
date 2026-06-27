#!/usr/bin/env python3
"""
precompute/build_reranker_scores.py

What it does:
  - Loads JD embedding (artifacts/jd_vec.npy), candidate embeddings (artifacts/cand_vecs.npy),
    and candidate IDs (artifacts/cand_ids.npy).
  - Performs fast ANN retrieval using cosine similarity (dot product of L2-normalized vectors)
    to retrieve the top 500 candidate IDs.
  - Loads the cross-encoder model 'cross-encoder/ms-marco-MiniLM-L-6-v2' using sentence-transformers.
  - Generates pairs of (jd_text, candidate_profile_text) for the top 500 candidates.
  - Scores the pairs on CPU.
  - Saves the results sorted by cross-encoder score descending to artifacts/reranker_scores.json.

Inputs:
  - artifacts/jd_vec.npy (shape [1, 384])
  - artifacts/cand_vecs.npy (shape [N, 384])
  - artifacts/cand_ids.npy (shape [N])
  - artifacts/features.parquet (for profile_text column)
  - jd_text.txt (source JD text)

Outputs:
  - artifacts/reranker_scores.json

Expected runtime:
  - ANN retrieval: < 1 second
  - Cross-encoder scoring: ~10-20 minutes on CPU for 500 pairs.
"""

import torch
from sentence_transformers import CrossEncoder
import json
from pathlib import Path
import numpy as np
import pandas as pd

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
JD_TEXT_PATH = Path(__file__).parent.parent / "jd_text.txt"


def main():
    torch.set_num_threads(4)
    print("Loading data...")
    # Load JD text
    if not JD_TEXT_PATH.exists():
        print(f"ERROR: cannot find {JD_TEXT_PATH}")
        import sys; sys.exit(1)
    jd_text = JD_TEXT_PATH.read_text(encoding="utf-8")

    # Load BGE vectors and IDs
    jd_vec = np.load(ARTIFACTS / "jd_vec.npy")        # [1, 384]
    cand_vecs = np.load(ARTIFACTS / "cand_vecs.npy")  # [N, 384]
    cand_ids = np.load(ARTIFACTS / "cand_ids.npy")    # [N]

    # Step 1: Fast ANN retrieval (dot product)
    print("Computing cosine similarities...")
    # Vectors are already L2 normalized, so dot product is cosine similarity
    similarities = np.dot(cand_vecs, jd_vec[0])
    
    # Get top 500 indices
    top_500_indices = np.argsort(-similarities)[:500]
    top_500_ids = cand_ids[top_500_indices]
    top_500_bge = similarities[top_500_indices]

    print(f"Top 500 retrieved. Cosine similarity range: {top_500_bge[0]:.4f} to {top_500_bge[-1]:.4f}")

    # Load features parquet to get profile_text for these 500 candidate IDs
    print("Loading profile texts from features.parquet...")
    df = pd.read_parquet(ARTIFACTS / "features.parquet", columns=["candidate_id", "profile_text"])
    # Filter to only the top 500
    df_filtered = df[df["candidate_id"].isin(top_500_ids)].set_index("candidate_id")

    # Build pairs
    pairs = []
    for cid in top_500_ids:
        # Fallback if profile_text doesn't exist
        profile_text = df_filtered.loc[cid, "profile_text"] if cid in df_filtered.index else ""
        pairs.append((jd_text, profile_text))

    # Step 2: Cross-encoder reranking
    print("Loading CrossEncoder model (cross-encoder/ms-marco-MiniLM-L-6-v2) on CPU...")
    # Ensure CPU only
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    print("Scoring 500 pairs with cross-encoder...")
    cross_scores = model.predict(pairs, show_progress_bar=True)

    # Compile results
    results = []
    for cid, bge_score, cross_score in zip(top_500_ids, top_500_bge, cross_scores):
        results.append({
            "candidate_id": str(cid),
            "bge_score": float(bge_score),
            "crossencoder_score": float(cross_score)
        })

    # Sort descending by cross-encoder score
    results.sort(key=lambda x: -x["crossencoder_score"])

    # Save
    out_path = ARTIFACTS / "reranker_scores.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved reranker scores to {out_path}")


if __name__ == "__main__":
    main()
