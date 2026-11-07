\
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prompts import build_shironaam_xlp_multigen_prompt
from src.providers import DecodingConfig, GeminiDirectGenerator
from src.utils import ensure_parent, first_k_paragraphs, resolve_column, sleep_for_rate_limit


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Shironaam second-benchmark experiment.")
    p.add_argument("--input", required=True, help="Shironaam test CSV")
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--max-output-tokens", type=int, default=600)
    p.add_argument("--rpm", type=float, default=0)
    p.add_argument("--limit", type=int)
    p.add_argument("--article-column")
    p.add_argument("--headline-column")
    p.add_argument("--category-column")
    p.add_argument("--tags-column")
    p.add_argument("--image-caption-column")
    args = p.parse_args()

    df = pd.read_csv(args.input).fillna("")
    if args.limit:
        df = df.head(args.limit).copy()
    article_col = resolve_column(df, args.article_column, ["article", "Article"])
    headline_col = resolve_column(df, args.headline_column, ["head_lines", "headline", "Headline", "Headlines"])
    category_col = resolve_column(df, args.category_column, ["category", "Category"])
    tags_col = resolve_column(df, args.tags_column, ["tags", "topic_words", "topic words"])
    caption_col = resolve_column(df, args.image_caption_column, ["image_caption", "caption", "Image Caption"])

    demo_path = Path(__file__).resolve().parents[1] / "config" / "shironaam_demonstrations.json"
    demonstrations = json.loads(demo_path.read_text(encoding="utf-8"))
    generator = GeminiDirectGenerator(
        "gemini-2.0-flash",
        DecodingConfig(args.temperature, args.top_p, args.max_output_tokens),
    )

    out_path = ensure_parent(args.output)
    processed = set()
    if out_path.exists():
        existing = pd.read_csv(out_path)
        if "source_row" in existing.columns:
            processed = set(existing["source_row"].astype(int).tolist())

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating Shironaam"):
        source_row = int(idx) if isinstance(idx, (int, float)) else 0
        if source_row in processed:
            continue
        start = time.time()
        article = first_k_paragraphs(row[article_col], 3)
        prompt = build_shironaam_xlp_multigen_prompt(
            article=article,
            category=str(row[category_col]),
            tags=str(row[tags_col]),
            image_caption=str(row[caption_col]),
            demonstrations=demonstrations,
            shots=3,
        )
        prediction = generator.generate(prompt)
        result = {
            "source_row": source_row,
            "reference_headline": str(row[headline_col]),
            "prompt": prompt,
            "prediction": prediction,
            "model": "gemini-2.0-flash",
            "language": "xlp",
            "variant": "multigen",
            "shots": 3,
            "context": "first_3_paragraphs",
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_output_tokens": args.max_output_tokens,
        }
        pd.DataFrame([result]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        sleep_for_rate_limit(start, args.rpm)


if __name__ == "__main__":
    main()
