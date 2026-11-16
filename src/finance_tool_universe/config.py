"""Single user configuration entry point."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def catalog_dir() -> Path:
    source = ROOT / "catalogs"
    installed = Path(sys.prefix) / "share" / "finance-tool-universe" / "catalogs"
    if (source / "tool_catalog.json").is_file():
        return source
    if (installed / "tool_catalog.json").is_file():
        return installed
    raise FileNotFoundError("Finance Tool Universe catalog files are missing")


def example_path() -> Path:
    source = ROOT / "config.example.yaml"
    installed = Path(sys.prefix) / "share" / "finance-tool-universe" / "config.example.yaml"
    if source.is_file():
        return source
    if installed.is_file():
        return installed
    raise FileNotFoundError("config.example.yaml is missing from the installation")


def config_path() -> Path:
    default = ROOT / "config.yaml" if (ROOT / "config.example.yaml").is_file() else Path.cwd() / "config.yaml"
    return Path(os.environ.get("FTU_CONFIG", default)).expanduser().resolve()


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; copy config.example.yaml to config.yaml and edit it")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def configured_path(value: str | None, default: str) -> Path:
    target = Path(value or default).expanduser()
    if not target.is_absolute():
        target = config_path().parent / target
    return target.resolve()


def apply_config(config: dict[str, Any]) -> None:
    """Export settings used by registered tools and their connectors."""
    credentials = config.get("credentials") or {}
    if not isinstance(credentials, dict):
        raise ValueError("credentials must be a mapping")
    for key, value in credentials.items():
        if value is None or value == "":
            continue
        if not str(key).isupper() or not str(key).replace("_", "").isalnum():
            raise ValueError(f"Invalid credential environment name: {key}")
        os.environ[str(key)] = str(value)

    network = config.get("network") or {}
    server = config.get("server") or {}
    os.environ["FTU_HOST"] = str(server.get("host") or "127.0.0.1")
    os.environ["FTU_PORT"] = str(server.get("port") or 8000)
    proxy = str(network.get("proxy_url") or "").strip()
    if proxy:
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            os.environ[key] = proxy
    os.environ["FTU_PROXY_URL"] = proxy
    os.environ["FTU_USER_AGENT"] = str(network.get("user_agent") or "")
    os.environ["FTU_ALLOWED_IPS"] = __import__("json").dumps(network.get("allowed_ips") or [])
    sources = config.get("sources") or {}
    os.environ["FTU_FMP_BASE_URL"] = str((sources.get("fmp") or {}).get("base_url") or "https://financialmodelingprep.com/stable")
    os.environ["FTU_MINERU_ENDPOINT"] = str((sources.get("mineru") or {}).get("base_url") or "https://mineru.net/api/pdf2json")

    paths = config.get("paths") or {}
    defaults = {
        "data_dir": "data",
        "output_dir": "output",
        "qlib_cn_data": "data/qlib/cn_data",
        "qlib_us_data": "data/qlib/us_data",
        "alphaqcm_repo": "external/AlphaQCM",
        "alphaforge_repo": "external/AlphaForge",
        "alphasage_repo": "external/AlphaSAGE",
        "qlib_benchmarks": "external/qlib_benchmark/benchmarks",
    }
    for key, default in defaults.items():
        os.environ[f"FTU_{key.upper()}"] = str(configured_path(paths.get(key), default))
    finbert = str(paths.get("finbert_model") or "").strip()
    os.environ["FTU_FINBERT_MODEL"] = str(configured_path(finbert, finbert)) if finbert else ""


def source_config(name: str) -> dict[str, Any]:
    return dict((load_config().get("sources") or {}).get(name) or {})
