"""Local files connector — read documents from local filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseConnector, register_connector


class LocalFilesConnector(BaseConnector):
    name = "local_files"
    actions = frozenset({"read", "search"})

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self._base_path = Path(self._config.get("base_path", "./data"))

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = self._base_path / p
        # Security: ensure resolved path stays under base_path
        try:
            p.resolve().relative_to(self._base_path.resolve())
        except ValueError:
            raise ValueError(f"Path traversal detected: {path}")
        return p

    async def fetch(self, action: str, **kwargs: Any) -> Any:
        dispatch = {
            "read": self._read,
            "search": self._search,
        }
        handler = dispatch.get(action)
        if handler is None:
            raise ValueError(f"LocalFiles: unsupported action '{action}'")
        return await handler(**kwargs)

    async def _read(self, path: str, format: str | None = None, **_: Any) -> dict:
        resolved = self._resolve(path)
        if not resolved.exists():
            return {"path": path, "format": format, "content": "", "pages": 0, "error": "File not found"}

        fmt = format or resolved.suffix.lstrip(".")

        try:
            if fmt in ("json",):
                content = json.loads(resolved.read_text(encoding="utf-8"))
                return {"path": path, "format": "json", "content": content, "pages": 1}

            elif fmt in ("csv",):
                import csv
                with open(resolved, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                return {"path": path, "format": "csv", "content": rows, "pages": 1}

            elif fmt in ("xlsx", "xls"):
                import openpyxl
                wb = openpyxl.load_workbook(resolved, read_only=True, data_only=True)
                sheets = {}
                for ws in wb.worksheets:
                    sheets[ws.title] = [
                        [cell.value for cell in row] for row in ws.iter_rows(max_row=1000)
                    ]
                wb.close()
                return {"path": path, "format": "xlsx", "content": sheets, "pages": len(sheets)}

            elif fmt in ("pdf",):
                try:
                    from PyPDF2 import PdfReader
                    reader = PdfReader(str(resolved))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages[:100])
                    return {"path": path, "format": "pdf", "content": text, "pages": len(reader.pages)}
                except ImportError:
                    return {"path": path, "format": "pdf", "content": "PyPDF2 not installed", "pages": 0}

            else:
                # Default: read as text
                content = resolved.read_text(encoding="utf-8", errors="replace")
                return {"path": path, "format": fmt or "text", "content": content[:50000], "pages": 1}

        except Exception as exc:
            return {"path": path, "format": fmt, "content": f"Error: {exc}", "pages": 0}

    async def _search(self, query: str, limit: int = 10, **_: Any) -> list[dict]:
        """Simple filename-based search within base_path."""
        if not self._base_path.exists():
            return []
        results = []
        for p in self._base_path.rglob("*"):
            if p.is_file() and query.lower() in p.name.lower():
                results.append({
                    "query": query,
                    "source": "local_files",
                    "data_type": "document",
                    "relevance": 1.0,
                    "payload": {"path": str(p), "name": p.name, "size": p.stat().st_size},
                })
                if len(results) >= limit:
                    break
        return results

    async def health_check(self) -> bool:
        return self._base_path.exists()


register_connector("local_files", LocalFilesConnector)
