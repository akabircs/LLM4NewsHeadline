\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tiktoken

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils import first_k_paragraphs, resolve_column


def main() -> None:
    p = argparse.ArgumentParser(description="Compute proxy input-token reduction using o200k_base.")
    p.add_argument("--input", required=True)
    p.add_argument("--article-column")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    df = pd.read_csv(args.input)
    article_col = resolve_column(df, args.article_column, ["Article", "article"])
    enc = tiktoken.get_encoding("o200k_base")
    full_counts, partial_counts, nparas = [], [], []
    for text in df[article_col].fillna("").astype(str):
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        full_counts.append(len(enc.encode(text)))
        partial_counts.append(len(enc.encode(first_k_paragraphs(text, args.k))))
        nparas.append(len(paragraphs))
    tmp = pd.DataFrame({"full": full_counts, "partial": partial_counts, "n_paragraphs": nparas})

    def summarize(sub: pd.DataFrame, label: str) -> dict:
        f, q = sub.full.to_numpy(), sub.partial.to_numpy()
        return {
            "subset": label,
            "n": len(sub),
            "full_mean": float(np.mean(f)),
            "full_sd": float(np.std(f, ddof=1)),
            "partial_mean": float(np.mean(q)),
            "partial_sd": float(np.std(q, ddof=1)),
            "reduction_percent": float(100 * (1 - np.mean(q) / np.mean(f))),
        }

    result = [summarize(tmp, "all")]
    result.append(summarize(tmp[tmp.n_paragraphs > args.k], f">{args.k}_paragraphs"))
    pd.DataFrame(result).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
