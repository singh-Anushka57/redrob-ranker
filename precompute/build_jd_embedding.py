#!/usr/bin/env python3
"""
precompute/build_jd_embedding.py
Run once. Produces artifacts/jd_vec.npy
"""
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

ARTIFACTS = Path(__file__).parent.parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
JD_TEXT_PATH = Path(__file__).parent.parent / "jd_text.txt"


def main():
    jd_text = JD_TEXT_PATH.read_text()
    print(f"JD text length: {len(jd_text)} chars")

    print("Loading BGE-small-en-v1.5 ...")
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    # BGE models expect a query prefix for asymmetric retrieval
    query = f"Represent this sentence for searching relevant passages: {jd_text}"
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)

    out_path = ARTIFACTS / "jd_vec.npy"
    np.save(out_path, vec)
    print(f"Saved JD embedding to {out_path}  shape={vec.shape}")


if __name__ == "__main__":
    main()
