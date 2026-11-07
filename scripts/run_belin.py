\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prompts import build_belin_prompt
from src.providers import DecodingConfig, make_generator
from src.utils import ensure_parent, first_k_paragraphs, resolve_column, sleep_for_rate_limit


def load_frame(path: str | None) -> pd.DataFrame:
    if path:
        return pd.read_csv(path)
    return load_dataset("osama39/BNHG", split="test").to_pandas()


def main() -> None:
    p = argparse.ArgumentParser(description="Run BeliN headline-generation experiments reported in the paper.")
    p.add_argument("--input", help="Optional local BeliN test CSV; otherwise osama39/BNHG test split is loaded.")
    p.add_argument("--output", required=True)
    p.add_argument("--provider", choices=["gemini", "llama", "gpt4o"], required=True)
    p.add_argument("--language", choices=["bnap", "xlp"], required=True)
    p.add_argument("--variant", choices=["baseline", "multigen"], required=True)
    p.add_argument("--shots", type=int, choices=[0, 1, 2, 3], default=0)
    p.add_argument("--context", default="full", help="full or an integer 1..5")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-output-tokens", type=int, default=50)
    p.add_argument("--rpm", type=float, default=0, help="Optional requests-per-minute throttle; 0 disables throttling.")
    p.add_argument("--limit", type=int, help="Optional row limit for smoke tests.")
    p.add_argument("--article-column")
    p.add_argument("--headline-column")
    p.add_argument("--sentiment-column")
    p.add_argument("--category-column")
    p.add_argument("--aspect-column")
    args = p.parse_args()

    context_k = None if args.context.lower() == "full" else int(args.context)
    if context_k is not None and context_k not in {1,2,3,4,5}:
        raise ValueError("--context must be full or 1..5")

    df = load_frame(args.input)
    if args.limit:
        df = df.head(args.limit).copy()

    article_col = resolve_column(df, args.article_column, ["Article", "article"])
    headline_col = resolve_column(df, args.headline_column, ["Headlines", "Headline", "head_lines", "headline"])
    sentiment_col = resolve_column(df, args.sentiment_column, ["Sentiment", "sentiment"])
    category_col = resolve_column(df, args.category_column, ["Category", "category"])
    aspect_col = resolve_column(df, args.aspect_column, ["Aspect", "aspect"])

    demo_path = Path(__file__).resolve().parents[1] / "config" / "belin_demonstrations.json"
    demonstrations = json.loads(demo_path.read_text(encoding="utf-8"))

    generator = make_generator(
        args.provider,
        DecodingConfig(args.temperature, args.top_p, args.max_output_tokens),
    )

    out_path = ensure_parent(args.output)
    processed = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "source_row" in existing.columns:
            processed = set(existing["source_row"].astype(int).tolist())

    rows = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
        source_row = int(idx) if isinstance(idx, (int, float)) else len(rows)
        if source_row in processed:
            continue
        start = time.time()
        article = first_k_paragraphs(row[article_col], context_k)
        prompt = build_belin_prompt(
            language=args.language,
            variant=args.variant,
            article=article,
            sentiment=str(row[sentiment_col]),
            category=str(row[category_col]),
            aspect=str(row[aspect_col]),
            demonstrations=demonstrations,
            shots=args.shots,
        )
        prediction = generator.generate(prompt)
        result = {
            "source_row": source_row,
            "reference_headline": str(row[headline_col]),
            "prompt": prompt,
            "prediction": prediction,
            "provider": args.provider,
            "language": args.language,
            "variant": args.variant,
            "shots": args.shots,
            "context": args.context,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
        }
        pd.DataFrame([result]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        sleep_for_rate_limit(start, args.rpm)


if __name__ == "__main__":
    main()
