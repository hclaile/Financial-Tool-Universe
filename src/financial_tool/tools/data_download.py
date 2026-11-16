"""Initialize a local Qlib market dataset using Qlib's official downloader."""

from __future__ import annotations

import os
from pathlib import Path


def run(args):
    market = str(getattr(args, "market", "cn") or "cn").lower()
    if market not in {"cn", "us"}:
        raise ValueError("market must be cn or us")
    region = str(getattr(args, "region", market) or market).lower()
    if region != market:
        raise ValueError("region must match market")
    if hasattr(args, "company"):
        raise ValueError("company price downloads use market_data_get_history")
    provider_uri = str(getattr(args, "provider_uri", "") or os.environ.get(
        f"FTU_QLIB_{market.upper()}_DATA", f"data/qlib/{market}_data"
    ))
    force = bool(getattr(args, "force", False))
    target = Path(provider_uri).expanduser()

    from qlib.tests.data import GetData, exists_qlib_data

    if exists_qlib_data(str(target)) and not force:
        return {"status": "exists", "market": market, "provider_uri": str(target)}
    target.mkdir(parents=True, exist_ok=True)
    GetData().qlib_data(
        name="qlib_data", target_dir=str(target), interval="1d", region=market,
        delete_old=force, exists_skip=not force,
    )
    if not exists_qlib_data(str(target)):
        raise RuntimeError(f"Qlib data download did not initialize {target}")
    return {"status": "downloaded", "market": market, "provider_uri": str(target)}
