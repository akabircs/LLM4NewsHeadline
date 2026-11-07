\
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd


def first_k_paragraphs(text: str, k: int | None) -> str:
    """Return the full document or the first k newline-delimited paragraphs."""
    text = "" if pd.isna(text) else str(text).strip()
    if k is None:
        return text
    paragraphs = [p.strip() for p in re.split(r"\n{1,}", text) if p.strip()]
    return "\n".join(paragraphs[:k])


def resolve_column(df: pd.DataFrame, requested: str | None, candidates: Iterable[str]) -> str:
    if requested:
        if requested not in df.columns:
            raise KeyError(f"Column '{requested}' was not found. Available columns: {list(df.columns)}")
        return requested
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(f"None of the expected columns {list(candidates)} were found. Available columns: {list(df.columns)}")


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def sleep_for_rate_limit(start_time: float, rpm: float | None) -> None:
    if not rpm or rpm <= 0:
        return
    interval = 60.0 / rpm
    remaining = interval - (time.time() - start_time)
    if remaining > 0:
        time.sleep(remaining)
