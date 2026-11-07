\
#!/usr/bin/env python3
"""Lightweight repository scan for common credential and identity-leak patterns.

This is a safety check, not a replacement for GitHub secret scanning or tools such as gitleaks.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

PATTERNS = {
    "huggingface_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "openai_key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "google_api_key": re.compile(r"AIza[A-Za-z0-9_-]{20,}"),
    "github_token": re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "email_address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "windows_user_path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\\\\\s]+", re.I),
    "unix_home_path": re.compile(r"/(?:home|Users)/[^/\s]+/"),
}

SKIP = {".git", ".venv", "venv", "__pycache__"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("root", nargs="?", default=".")
    args = p.parse_args()
    root = Path(args.root)
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".zip", ".pyc"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                # Do not print credential-like values themselves.
                line = text.count("\n", 0, match.start()) + 1
                findings.append((str(path), line, label))
    if findings:
        print("Potential sensitive strings detected:")
        for path, line, label in findings:
            print(f"  {path}:{line}: {label}")
        return 1
    print("No matching credentials, email addresses, or user-home paths detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
