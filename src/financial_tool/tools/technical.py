"""Technical analysis tools backed by regional OHLCV failover."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..compat.metrics import compute_technical_indicator_frame, compute_technical_indicators

from ..connectors.base import fetch_with_failover
from ..errors import DataValidationError
from ..symbols import classify_instrument
from ._index_history import fetch_index_history


def _pattern_events(frame: pd.DataFrame, patterns: str) -> pd.DataFrame:
    columns = ["date", "pattern", "signal"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    requested = str(patterns or "all").strip().lower()
    events: list[dict[str, Any]] = []
    cross = (frame["sma_5"] - frame["sma_20"]).apply(lambda value: 1 if value > 0 else -1 if value < 0 else 0)
    for index in range(1, len(frame)):
        date = frame.iloc[index].get("date")
        if requested in {"all", "moving_average", "ma_cross"}:
            if cross.iloc[index - 1] <= 0 < cross.iloc[index]:
                events.append({"date": date, "pattern": "golden_cross", "signal": "bullish"})
            elif cross.iloc[index - 1] >= 0 > cross.iloc[index]:
                events.append({"date": date, "pattern": "death_cross", "signal": "bearish"})
        rsi = frame.iloc[index].get("rsi_14")
        prior_rsi = frame.iloc[index - 1].get("rsi_14")
        if requested in {"all", "rsi"} and pd.notna(rsi) and pd.notna(prior_rsi):
            if prior_rsi < 70 <= rsi:
                events.append({"date": date, "pattern": "rsi_overbought", "signal": "bearish"})
            elif prior_rsi > 30 >= rsi:
                events.append({"date": date, "pattern": "rsi_oversold", "signal": "bullish"})
    return pd.DataFrame(events, columns=columns)


def _write_technical_csvs(
    frame: pd.DataFrame,
    *,
    input_data_path: str | None,
    output_dir: str | None,
    patterns: str,
) -> tuple[Path, Path]:
    if output_dir:
        target_dir = Path(output_dir).resolve()
    elif input_data_path:
        target_dir = Path(input_data_path).resolve().parent
    else:
        target_dir = Path(tempfile.mkdtemp(prefix="finskill-technical-"))
    target_dir.mkdir(parents=True, exist_ok=True)
    indicators_path = target_dir / "technical_indicators.csv"
    events_path = target_dir / "pattern_events.csv"
    frame.to_csv(indicators_path, index=False)
    _pattern_events(frame, patterns).to_csv(events_path, index=False)
    return indicators_path, events_path


def _artifact_rows(payload: Any, symbol: str | None = None) -> list[dict[str, Any]]:
    target_symbol = str(symbol or "").strip().upper()

    def keep(row: dict[str, Any]) -> bool:
        if not target_symbol:
            return True
        row_symbol = str(row.get("symbol") or "").strip().upper()
        return not row_symbol or row_symbol == target_symbol

    if isinstance(payload, list):
        return [dict(item) for item in payload if isinstance(item, dict) and keep(item)]
    if not isinstance(payload, dict):
        return []

    rows_by_symbol = payload.get("rows_by_symbol")
    if isinstance(rows_by_symbol, dict):
        if target_symbol:
            rows = _artifact_rows(rows_by_symbol.get(target_symbol), symbol)
            if rows:
                return rows
            for key, nested in rows_by_symbol.items():
                if str(key).strip().upper() == target_symbol:
                    return _artifact_rows(nested, symbol)
        rows: list[dict[str, Any]] = []
        for nested in rows_by_symbol.values():
            rows.extend(_artifact_rows(nested, symbol))
        if rows:
            return rows

    for key in ("rows", "data", "result", "series"):
        rows = _artifact_rows(payload.get(key), symbol)
        if rows:
            return rows

    for item in payload.get("tool_results", []) if isinstance(payload.get("tool_results"), list) else []:
        if isinstance(item, dict):
            rows = _artifact_rows(item.get("result"), symbol)
            if rows:
                return rows

    if target_symbol:
        for key, nested in payload.items():
            if str(key).strip().upper() == target_symbol:
                return _artifact_rows(nested, symbol)

    return []


def _load_input_artifact(path_value: str, symbol: str | None = None) -> tuple[pd.DataFrame, str]:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise DataValidationError("input_data_path", "must reference an existing CSV or JSON artifact")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if symbol and "symbol" in frame.columns:
            frame = frame[
                frame["symbol"].astype(str).str.upper() == str(symbol).upper()
            ].copy()
        return frame, "artifact_csv"
    if suffix != ".json":
        raise DataValidationError("input_data_path", "must reference a CSV or JSON artifact")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataValidationError("input_data_path", f"invalid JSON artifact: {exc.msg}") from exc
    rows = _artifact_rows(payload, symbol)
    if not rows:
        raise DataValidationError("input_data_path", "JSON artifact contains no OHLCV rows")
    return pd.DataFrame(rows), "artifact_json"


def _select_output_frame(
    frame: pd.DataFrame,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return frame
    dates = pd.to_datetime(frame["date"], errors="coerce")
    mask = pd.Series(True, index=frame.index)
    if start_date:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date:
        mask &= dates <= pd.Timestamp(end_date)
    return frame.loc[mask].copy()


def _normalize_ohlcv_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    if "date" in normalized:
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    required = [column for column in ("date", "close") if column in normalized]
    if len(required) == 2:
        normalized = normalized.dropna(subset=required).sort_values("date").reset_index(drop=True)
    return normalized


def _series_payload(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    available = [column for column in columns if column in frame.columns]
    records: list[dict[str, Any]] = []
    for row in frame[available].to_dict(orient="records"):
        record: dict[str, Any] = {}
        for key, value in row.items():
            if pd.isna(value):
                record[key] = None
            elif key in {"date", "symbol"}:
                record[key] = str(value)
            else:
                try:
                    record[key] = float(value)
                except (TypeError, ValueError):
                    record[key] = str(value)
        records.append(record)
    return records


def register(mcp) -> None:

    @mcp.tool()
    async def talib_technical_indicators(
        input_path: str, output_path: str | None = None,
        indicators: list[str] | None = None, config: dict[str, Any] | None = None,
        validate: bool = False,
    ) -> dict:
        from types import SimpleNamespace
        from ._blocking import run_blocking

        def execute():
            from . import talib_technical_indicators as engine
            return engine.run(SimpleNamespace(
                input_path=input_path, output_path=output_path, indicators=indicators,
                config=config or {}, validate=validate,
            ))
        return await run_blocking(execute, name="talib_technical_indicators", output_path=output_path)

    @mcp.tool()
    async def technical_detect_patterns(
        run_dir: str | None = None,
        symbol: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        lookback: int = 14,
        patterns: str = "all",
        window: int = 10,
        region: str = "US",
        input_csv_path: str | None = None,
        input_data_path: str | None = None,
        input_json_path: str | None = None,
        sma_windows: list[int] | None = None,
        ema_windows: list[int] | None = None,
        output_dir: str | None = None,
    ) -> Any:
        if not symbol:
            return {
                "status": "invalid_request",
                "reason": "symbol is required",
            }
        instrument = classify_instrument(symbol)
        if instrument.kind == "macro_series":
            raise DataValidationError(
                "symbol",
                f"{symbol} is macro data; call macro_get_series("
                f"series_id={instrument.provider_symbol!r}, region='US')",
            )
        if instrument.kind == "shipping_series":
            raise DataValidationError(
                "symbol",
                f"{symbol} is a shipping index; call a shipping-series tool "
                f"with series_id={instrument.provider_symbol!r}",
            )
        symbol = instrument.provider_symbol
        region = instrument.region or region
        input_paths = [path for path in (input_data_path, input_csv_path, input_json_path) if path]
        if len(input_paths) > 1:
            raise DataValidationError("input_data_path", "provide only one input artifact path")
        artifact_path = input_paths[0] if input_paths else None
        if artifact_path:
            frame, source = _load_input_artifact(artifact_path, symbol=symbol)
        else:
            if instrument.kind == "index":
                history = await fetch_index_history(
                    fetch_with_failover,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    region=region,
                )
                if history is None:
                    history = await fetch_with_failover(
                        "ohlcv",
                        "ohlcv",
                        region=region,
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval="1d",
                    )
            else:
                history = await fetch_with_failover(
                    "ohlcv",
                    "ohlcv",
                    region=region,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval="1d",
                )
            rows = history.get("rows", []) if isinstance(history, dict) else history
            frame = pd.DataFrame(rows or [])
            source = "regional_ohlcv_failover"
        frame = _normalize_ohlcv_frame(frame)
        if sma_windows:
            resolved_sma = list(sma_windows)
        else:
            # `lookback` is the default fast-moving-average window so the public
            # parameter has an effect (sma_5/sma_20 remain derived constants).
            resolved_sma = [5, 20, lookback]
        indicator_frame = compute_technical_indicator_frame(
            frame,
            sma_windows=resolved_sma,
            ema_windows=ema_windows,
        )
        if indicator_frame.empty:
            return {
                "status": "insufficient_data",
                "symbol": symbol,
                "observations": int(len(frame)),
                "source": source,
            }
        output_frame = _select_output_frame(indicator_frame, start_date, end_date)
        indicators_path, events_path = _write_technical_csvs(
            output_frame,
            input_data_path=artifact_path,
            output_dir=output_dir or run_dir,
            patterns=patterns,
        )
        base_series_columns = [
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "return_1d",
            "return_5d",
            "return_20d",
            "volatility_20d",
            "drawdown",
            "macd",
            "macd_signal",
            "macd_hist",
            "rsi_14",
            "bollinger_mid_20",
            "bollinger_upper_20",
            "bollinger_lower_20",
            "atr_14",
            "volume_zscore_20",
        ]
        moving_average_columns = [
            column
            for column in output_frame.columns
            if column.startswith(("sma_", "ema_"))
        ]
        indicator_columns = [*base_series_columns, *moving_average_columns]
        produced_columns = list(
            dict.fromkeys(column for column in indicator_columns if column in output_frame.columns)
        )
        return {
            "status": "success",
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "lookback": lookback,
            "patterns": patterns,
            "window": window,
            "observations": int(len(output_frame)),
            "source": source,
            "indicators_path": str(indicators_path),
            "pattern_events_path": str(events_path),
            "produced_columns": produced_columns,
            "series": _series_payload(output_frame, produced_columns),
            "summary": compute_technical_indicators(indicator_frame),
            "run_dir": run_dir,
        }
