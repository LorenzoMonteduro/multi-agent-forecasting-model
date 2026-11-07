"""Parse the JSON object that ends a model reply, and URLs in replies."""
import json
import re

URL_RE = re.compile(r"https?://[^\s\)\]\}>\"'`,]+")


def _objects(text: str) -> list[str]:
    """Top-level balanced {...} blocks, ignoring braces inside JSON strings."""
    out, depth, start, in_str, esc = [], 0, -1, False, False
    for i, ch in enumerate(text):
        if in_str:
            esc, in_str = (False, in_str) if esc else (ch == "\\", ch != '"')
        elif ch == '"' and depth:
            in_str = True
        elif ch == "{":
            start = i if depth == 0 else start
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out


def last_json_object(text: str | None) -> dict | None:
    for chunk in reversed(_objects(text or "")):
        for cand in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):   # tolerate trailing commas
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def ends_with_json_line(text: str | None) -> bool:
    """True if the last non-empty line (code fences stripped) is a JSON object."""
    lines = [l for l in (l.strip().strip("`").strip() for l in (text or "").splitlines()) if l and l.lower() != "json"]
    try:
        return bool(lines) and isinstance(json.loads(lines[-1]), dict)
    except json.JSONDecodeError:
        return False


def to_number(v) -> float | None:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?", v.replace(",", ""))
        return float(m.group()) if m else None
    return None


def extract_urls(text: str | None) -> list[str]:
    return list(dict.fromkeys(u.rstrip(".;:") for u in URL_RE.findall(text or "")))
