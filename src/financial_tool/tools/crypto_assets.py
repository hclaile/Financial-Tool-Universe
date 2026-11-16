"""List FMP cryptocurrency assets not exposed by the price tools."""

from __future__ import annotations

import json
import os
from pathlib import Path

import requests


def run(args):
    output_path = str(getattr(args, "output_path", "") or "").strip()
    if not output_path:
        raise ValueError("output_path is required")
    limit = int(getattr(args, "limit", 100) or 100)
    if limit < 1:
        raise ValueError("limit must be positive")
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        raise ValueError("FMP API key is missing; set credentials.FMP_API_KEY in config.yaml")
    base = os.environ.get("FTU_FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/")
    response = requests.get(
        f"{base}/cryptocurrency-list", params={"apikey": api_key}, timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("FMP cryptocurrency list response must be a list")
    data = data[:limit]
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"count": len(data), "output_path": str(output), "data": data}
