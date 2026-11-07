\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import ensure_parent, resolve_column


def split_paragraphs(text: str) -> list[str]:
    import re
    return [p.strip() for p in re.split(r"\n{1,}", str(text).strip()) if p.strip()]


def load_frame(dataset_name: str, input_path: str | None, split: str) -> pd.DataFrame:
    if input_path:
        return pd.read_csv(input_path)
    if dataset_name != "belin":
        raise ValueError("A local --input CSV is required for Shironaam.")
    return load_dataset("osama39/BNHG", split=split).to_pandas()


def main() -> None:
    p = argparse.ArgumentParser(description="Reference-headline/paragraph semantic-alignment analysis (Algorithm 1).")
    p.add_argument("--dataset", choices=["belin", "shironaam"], required=True)
    p.add_argument("--input", help="Local CSV. Optional for BeliN; required for Shironaam.")
    p.add_argument("--split", default="train")
    p.add_argument("--article-column")
    p.add_argument("--headline-column")
    p.add_argument("--embedding-model", default="banglagov/banBERT-Base")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--limit", type=int)
    args = p.parse_args()

    df = load_frame(args.dataset, args.input, args.split)
    if args.limit:
        df = df.head(args.limit).copy()
    article_col = resolve_column(df, args.article_column, ["Article", "article"])
    headline_col = resolve_column(df, args.headline_column, ["Headlines", "Headline", "head_lines", "headline"])

    model = SentenceTransformer(args.embedding_model)
    records = []
    all_scores: dict[int, list[float]] = {i: [] for i in range(1, 8)}
    all_scores[8] = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Semantic alignment"):
        paragraphs = split_paragraphs(row[article_col])
        if not paragraphs:
            continue
        headline = str(row[headline_col])
        h = model.encode(headline, convert_to_tensor=True)
        p_emb = model.encode(paragraphs, convert_to_tensor=True)
        if p_emb.dim() == 1:
            p_emb = p_emb.unsqueeze(0)
        scores = cos_sim(p_emb, h.unsqueeze(0)).squeeze().detach().cpu().numpy()
        scores = np.atleast_1d(scores).astype(float)
        best_zero = int(np.argmax(scores))
        best_one = best_zero + 1
        records.append({
            "source_row": idx,
            "best_paragraph_position": best_one,
            "best_similarity": float(scores[best_zero]),
            "n_paragraphs": len(paragraphs),
        })
        for pos, score in enumerate(scores, start=1):
            group = pos if pos <= 7 else 8
            all_scores[group].append(float(score))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = pd.DataFrame(records)
    rec.to_csv(out_dir / "alignment_rows.csv", index=False)

    total = len(rec)
    rows = []
    cumulative = 0
    for label, selector in [
        ("1st", rec.best_paragraph_position == 1),
        ("2nd", rec.best_paragraph_position == 2),
        ("3rd", rec.best_paragraph_position == 3),
        ("4th", rec.best_paragraph_position == 4),
        ("5th", rec.best_paragraph_position == 5),
        ("6th", rec.best_paragraph_position == 6),
        ("7th", rec.best_paragraph_position == 7),
        (">=8th", rec.best_paragraph_position >= 8),
    ]:
        n = int(selector.sum())
        cumulative += n
        rows.append({
            "paragraph": label,
            "n": n,
            "percent": 100.0 * n / total if total else np.nan,
            "cumulative_n": cumulative,
            "cumulative_percent": 100.0 * cumulative / total if total else np.nan,
        })
    pd.DataFrame(rows).to_csv(out_dir / "position_distribution.csv", index=False)

    # Save long-form paragraph scores for reproducing the paper/supplement boxplots.
    score_rows = []
    for group, values in all_scores.items():
        label = str(group) if group < 8 else ">=8"
        score_rows.extend({"paragraph_group": label, "cosine_similarity": v} for v in values)
    pd.DataFrame(score_rows).to_csv(out_dir / "paragraph_similarity_scores.csv", index=False)


if __name__ == "__main__":
    main()
