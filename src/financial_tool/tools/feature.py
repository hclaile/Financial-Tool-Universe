import os
import json
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import yaml


def _load_config(path: str) -> Dict[str, Any]:
	path = os.path.expanduser(path)
	with open(path, "r", encoding="utf-8") as f:
		if path.endswith((".yml", ".yaml")):
			return yaml.safe_load(f)
		return json.load(f)


def _init_qlib_from_cfg(cfg: Dict[str, Any]) -> None:
	import qlib
	from qlib.config import REG_CN, REG_US

	provider_uri = os.path.expanduser(cfg.get("provider_uri", os.environ.get("FTU_QLIB_CN_DATA", "data/qlib/cn_data")))
	region = cfg.get("region", "cn")
	region_const = REG_CN if region == "cn" else REG_US
	qlib.init(provider_uri=provider_uri, region=region_const)


def _build_dataset(cfg: Dict[str, Any]) -> pd.DataFrame:
	"""Prioritize reading from raw_parquet; otherwise use qlib DataHandler workflow."""
	raw_path = cfg.get("raw_parquet")
	if raw_path:
		raw_path = os.path.expanduser(raw_path)
		return pd.read_parquet(raw_path)

	from qlib.contrib.data.dataset import DatasetH
	from qlib.contrib.data.handler import DataHandlerLP

	handler_config = {
		"start_time": cfg.get("start_time"),
		"end_time": cfg.get("end_time"),
		"instruments": cfg.get("instruments", "csi300"),
		"fields": cfg.get("features", ["$close", "$open"]),
		"filter_pipe": cfg.get("filters", []),
	}
	label = cfg.get("label")
	if label:
		handler_config["labels"] = [label]

	dataset = DatasetH(handler=DataHandlerLP, handler_kwargs=handler_config)
	df = dataset.handler.fetch()
	return df


def run(args) -> None:
	cfg = _load_config(args.config)
	# raw_parquet Mode not required init qlib
	if not cfg.get("raw_parquet"):
		_init_qlib_from_cfg(cfg)
	df = _build_dataset(cfg)
	out = os.path.expanduser(args.output)
	Path(os.path.dirname(out) or ".").mkdir(parents=True, exist_ok=True)
	if out.endswith(".parquet"):
		df.to_parquet(out)
	else:
		df.to_csv(out)
	print(f"Features exported: {out}")
