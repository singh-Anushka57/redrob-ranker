#!/usr/bin/env python3
"""
precompute/build_embeddings.py
Embeds all 100K candidates using BGE-small-en-v1.5 (CPU friendly, ~133MB).
Run once. Produces:
  artifacts/cand_vecs.npy   shape [N, 384]
  artifacts/cand_ids.npy    shape [N]  (string IDs in matching order)
"""
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


ARTIFACTS = Path(__file__).parent.parent / "artifacts"
BATCH_SIZE = 512


def main():
    torch.set_num_threads(4)
    feat_path = ARTIFACTS / "features.parquet"
    if not feat_path.exists():
        print("ERROR: run build_features.py first")
        import sys; sys.exit(1)

    df = pd.read_parquet(feat_path, columns=["candidate_id", "profile_text"])
    print(f"Embedding {len(df)} candidates ...")

    model = SentenceTransformer("BAAI/bge-small-en-v1.5")

    texts = df["profile_text"].tolist()
    ids = df["candidate_id"].tolist()

    all_vecs = []
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Batches"):
        batch = texts[i : i + BATCH_SIZE]
        vecs = model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
        all_vecs.append(vecs)

    cand_vecs = np.vstack(all_vecs).astype(np.float32)
    cand_ids = np.array(ids)

    np.save(ARTIFACTS / "cand_vecs.npy", cand_vecs)
    np.save(ARTIFACTS / "cand_ids.npy", cand_ids)
    print(f"Saved cand_vecs.npy {cand_vecs.shape} and cand_ids.npy {cand_ids.shape}")


if __name__ == "__main__":
    main()
