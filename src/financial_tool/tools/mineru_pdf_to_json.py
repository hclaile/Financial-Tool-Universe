"""
Use MinerU API to convert PDF to JSON structured data

Default uses environment variable MINERU_API_KEY as authentication token, can be overridden by parameter.
"""
import os
import sys
import json
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

import requests


def _print(*args, **kwargs):
    """Print to stderr to avoid interfering with RPC response"""
    print(*args, file=sys.stderr, **kwargs)


DEFAULT_ENDPOINT = os.environ.get("FTU_MINERU_ENDPOINT", "https://mineru.net/api/pdf2json")


def _ensure_dir(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def run(args: SimpleNamespace):
    """
    Run PDF to JSON processing flow.

    Parameters(SimpleNamespace):
    - pdf_path: Local PDF file path (required))
    - output_path: Output JSON path, default is same directory as pdf, same name plus .json
    - api_key: MinerU API Key,Optional, default reads from environment variable MINERU_API_KEY
    - endpoint: MinerU API Endpoint, optional, default https://mineru.net/api/pdf2json
    """
    pdf_path = Path(os.path.expanduser(getattr(args, "pdf_path", "")))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF File does not exist: {pdf_path}")

    output_path = getattr(args, "output_path", None)
    if not output_path:
        output_path = pdf_path.with_suffix(".json")
    output_path = Path(os.path.expanduser(output_path))

    endpoint = getattr(args, "endpoint", DEFAULT_ENDPOINT)
    api_key: Optional[str] = getattr(args, "api_key", None) or os.getenv("MINERU_API_KEY")

    if not api_key:
        raise ValueError("Missing api_key: please pass parameter api_key or set environment variable MINERU_API_KEY")

    headers = {"Authorization": f"Bearer {api_key}"}

    _print(f"Calling MinerU: {endpoint}")
    _print(f"Input: {pdf_path}")
    _print(f"Output: {output_path}")

    with pdf_path.open("rb") as f:
        files = {"file": ("document.pdf", f, "application/pdf")}
        resp = requests.post(endpoint, headers=headers, files=files, timeout=300)

    try:
        resp.raise_for_status()
    except Exception:
        _print(f"Request failed: status={resp.status_code}, text={resp.text}")
        raise

    try:
        data = resp.json()
    except Exception:
        _print("Response is not JSON, raw text:")
        _print(resp.text[:500])
        raise ValueError("MinerU Response cannot be parsed as JSON")

    _ensure_dir(output_path)
    with output_path.open("w", encoding="utf-8") as out_f:
        json.dump(data, out_f, ensure_ascii=False, indent=2)

    _print("✓ Conversion completed")
    return {
        "status": "success",
        "endpoint": endpoint,
        "output_path": str(output_path),
        "bytes_written": output_path.stat().st_size if output_path.exists() else 0,
    }
