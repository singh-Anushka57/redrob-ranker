#!/usr/bin/env python3
"""
precompute/run_precompute.py

Orchestrates all offline precomputations, behavioral modeling, fast ranking, and submission validation sequentially.
"""

import subprocess
import sys
import time
from pathlib import Path

# Use the current environment's Python interpreter
PYTHON_EXE = sys.executable
ROOT_DIR = Path(__file__).parent.parent


def run_command(cmd_args, desc):
    print(f"\n========================================\n[RUNNING] {desc}\nCommand: {' '.join(cmd_args)}\n========================================")
    t0 = time.time()
    # Execute the command with output streamed to stdout/stderr
    res = subprocess.run(cmd_args, cwd=ROOT_DIR)
    t1 = time.time()
    if res.returncode != 0:
        print(f"\n[ERROR] {desc} failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[COMPLETED] {desc} in {t1 - t0:.2f} seconds.")


def main():
    t_start = time.time()
    print("Starting master precompute and ranking orchestrator...")
    
    # 1. Feature extraction
    run_command([PYTHON_EXE, "precompute/build_features.py", "candidates.jsonl"], "1. Building candidate features Parquet")
    
    # 2. JD Vector Embedding
    run_command([PYTHON_EXE, "precompute/build_jd_embedding.py"], "2. Building Job Description BGE embedding")
    
    # 3. Candidate Embeddings (CPU-bound, takes time)
    run_command([PYTHON_EXE, "precompute/build_embeddings.py"], "3. Generating BGE embeddings for 100K profiles")
    
    # 4. Honeypot Flagging
    run_command([PYTHON_EXE, "precompute/flag_honeypots.py", "candidates.jsonl"], "4. Identifying honeypot candidate profiles")
    
    # 5. Cosine + Cross-Encoder Reranking
    run_command([PYTHON_EXE, "precompute/build_reranker_scores.py"], "5. Running ANN retrieval & Cross-Encoder reranking")
    
    # 6. Behavioral Scoring
    run_command([PYTHON_EXE, "precompute/build_behavioral_scores.py"], "6. Scoring conversion probabilities (Stage 2)")
    
    # 7. Grounded Reasoning Generation
    run_command([PYTHON_EXE, "precompute/generate_reasoning.py", "--candidates", "candidates.jsonl"], "7. Generating cached reasoning strings")
    
    # 8. Timed Step (Final submission CSV)
    run_command([PYTHON_EXE, "rank.py", "--candidates", "candidates.jsonl", "--out", "team_submission.csv"], "8. Executing timed ranker")
    
    # 9. Format Validation
    run_command([PYTHON_EXE, "validate_submission.py", "team_submission.csv"], "9. Validating final submission file structure")
    
    t_total = time.time() - t_start
    print(f"\n========================================\nALL STEPS COMPLETED SUCCESSFULLY!\nTotal execution time: {t_total / 60.0:.2f} minutes.\nSaved final CSV to team_submission.csv\n========================================")


if __name__ == "__main__":
    main()
