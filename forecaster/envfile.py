"""Minimal .env loader (KEY=value lines); values are never printed and never override the real environment."""
from __future__ import annotations

import os
from pathlib import Path


def parse_env(text: str) -> dict[str, str]:
    out = {}
    for line in (l.strip() for l in text.splitlines()):
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.removeprefix("export ").split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        if k:
            out[k] = v
    return out


def load_env(path: Path) -> list[str]:
    """Set missing variables from `path` (UTF-8, UTF-8 with BOM or UTF-16). Returns the keys found."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    pairs = parse_env(raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig"))
    for k, v in pairs.items():
        if not os.environ.get(k):
            os.environ[k] = v
    return sorted(pairs)
