"""Network defaults loaded before external connector modules are imported."""

from __future__ import annotations

import os


DEFAULT_PROXY_URL = ""
PROXY_URL = os.getenv("FTU_PROXY_URL", "")
PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else {}
DEFAULT_USER_AGENT = ""


def _ensure_no_proxy() -> None:
    values = [
        item.strip()
        for item in (os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or "").split(",")
        if item.strip()
    ]
    seen = {item.lower() for item in values}
    for item in ("127.0.0.1", "localhost"):
        if item not in seen:
            values.append(item)
            seen.add(item)
    joined = ",".join(values)
    os.environ["NO_PROXY"] = joined
    os.environ["no_proxy"] = joined


def configure_network_env() -> dict[str, str]:
    """Set proxy environment variables without overwriting deployment overrides."""

    global PROXY_URL, PROXIES
    PROXY_URL = os.getenv("FTU_PROXY_URL", DEFAULT_PROXY_URL)
    if PROXY_URL:
        os.environ["HTTP_PROXY"] = PROXY_URL
        os.environ["HTTPS_PROXY"] = PROXY_URL
        os.environ["http_proxy"] = PROXY_URL
        os.environ["https_proxy"] = PROXY_URL
    _ensure_no_proxy()
    PROXIES = {"http": PROXY_URL, "https": PROXY_URL} if PROXY_URL else {}
    return dict(PROXIES)


def sec_user_agent(app_name: str | None = None, contact_email: str | None = None) -> str:
    """Return the stable application identity used by network connectors."""

    del app_name, contact_email
    return os.getenv("FTU_USER_AGENT", "")


configure_network_env()
