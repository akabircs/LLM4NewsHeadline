\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import nltk
import numpy as np
import pandas as pd
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import single_meteor_score
from scipy.stats import bootstrap
from tqdm import tqdm

from multilingual_rouge import rouge_scorer
from bengali_stemmer.rafikamal2014 import RafiStemmer


def basic_tokenizer(sentence: str) -> list[str]:
    import re
    splitter = re.compile(r'([.,!?"\':;)(])')
    tokens = []
    for fragment in sentence.strip().split():
        tokens.extend(re.split(splitter, fragment))
    return [t for t in tokens if t]


def bca_ci(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    res = bootstrap((values,), np.mean, vectorized=False, n_resamples=10000, method="BCa", random_state=42)
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def main() -> None:
    p = argparse.ArgumentParser(description="Compute automatic headline-generation metrics and BCa confidence intervals.")
    p.add_argument("--input", required=True)
    p.add_argument("--reference-column", default="reference_headline")
    p.add_argument("--prediction-column", default="prediction")
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    nltk.download("wordnet", quiet=True)
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    df = pd.read_csv(args.input)
    refs = df[args.reference_column].fillna("").astype(str).tolist()
    preds = df[args.prediction_column].fillna("").astype(str).tolist()

    # BERTScore in one batch to preserve the same language setting while avoiding per-row model reloads.
    _, _, bert_f1 = bert_score(preds, refs, lang="bn", verbose=True)
    bert_vals = bert_f1.detach().cpu().numpy().astype(float)

    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        callable_stemmer=RafiStemmer().stem_word,
        use_stemmer=True,
        lang="bengali",
    )

    rows = []
    for i, (ref, pred) in enumerate(tqdm(zip(refs, preds), total=len(refs), desc="Metrics")):
        ref_basic = basic_tokenizer(ref)
        pred_basic = basic_tokenizer(pred)
        bleu = sentence_bleu([ref_basic], pred_basic, weights=[1, 0, 0, 0])
        meteor = single_meteor_score(word_tokenize(ref), word_tokenize(pred))
        r = rouge.score(ref, pred)
        rows.append({
            "source_row": df.iloc[i]["source_row"] if "source_row" in df.columns else i,
            "BLEU": float(bleu),
            "METEOR": float(meteor),
            "BERTScore_F1": float(bert_vals[i]),
            "ROUGE1_F": float(r["rouge1"].fmeasure),
            "ROUGE2_F": float(r["rouge2"].fmeasure),
            "ROUGEL_F": float(r["rougeL"].fmeasure),
        })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inst = pd.DataFrame(rows)
    inst.to_csv(out_dir / "instance_scores.csv", index=False)

    summaries = []
    for metric in ["BLEU", "METEOR", "BERTScore_F1", "ROUGE1_F", "ROUGE2_F", "ROUGEL_F"]:
        v = inst[metric].to_numpy(dtype=float)
        low, high = bca_ci(v)
        summaries.append({
            "metric": metric,
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "bca95_low": low,
            "bca95_high": high,
            "n": len(v),
        })
    pd.DataFrame(summaries).to_csv(out_dir / "summary_metrics.csv", index=False)


if __name__ == "__main__":
    main()
