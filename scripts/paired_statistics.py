\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
from scipy.stats import bootstrap, wilcoxon


def main() -> None:
    p = argparse.ArgumentParser(description="Paired mean difference, 95% BCa CI, and two-sided Wilcoxon signed-rank test.")
    p.add_argument("--a", required=True, help="First instance_scores.csv")
    p.add_argument("--b", required=True, help="Second instance_scores.csv")
    p.add_argument("--metric", default="BERTScore_F1")
    p.add_argument("--id-column", default="source_row")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    a = pd.read_csv(args.a)[[args.id_column, args.metric]].rename(columns={args.metric: "a"})
    b = pd.read_csv(args.b)[[args.id_column, args.metric]].rename(columns={args.metric: "b"})
    m = a.merge(b, on=args.id_column, how="inner").dropna()
    d = (m.a - m.b).to_numpy(dtype=float)
    if len(d) < 2:
        raise ValueError("At least two paired observations are required")
    ci = bootstrap((d,), np.mean, vectorized=False, n_resamples=10000, method="BCa", random_state=42).confidence_interval
    if np.allclose(d, 0):
        p_value = 1.0
        statistic = 0.0
    else:
        w = wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        statistic, p_value = float(w.statistic), float(w.pvalue)
    result = {
        "metric": args.metric,
        "n_pairs": int(len(d)),
        "mean_difference_a_minus_b": float(np.mean(d)),
        "bca95_low": float(ci.low),
        "bca95_high": float(ci.high),
        "wilcoxon_statistic": statistic,
        "wilcoxon_p": p_value,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
