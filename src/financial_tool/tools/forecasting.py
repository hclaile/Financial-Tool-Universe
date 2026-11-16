"""Time series and text model inference tools."""

from __future__ import annotations

from types import SimpleNamespace

from ._blocking import run_blocking


def register(mcp) -> None:
    @mcp.tool()
    async def time_series_forecast(
        input_path: str, output_path: str | None = None,
        datetime_column: str = "date", value_column: str = "close",
        instrument_column: str | None = None, instrument: str | None = None,
        frequency: str | None = None, model_name: str = "time-series-transformer",
        lookback_window: int = 60, forecast_steps: int = 10,
        device: str = "auto",
    ) -> dict:
        def execute():
            from . import time_series_forecast as engine
            return engine.run(SimpleNamespace(
                input_path=input_path, output_path=output_path,
                datetime_column=datetime_column, value_column=value_column,
                instrument_column=instrument_column, instrument=instrument,
                frequency=frequency, model_name=model_name,
                lookback_window=lookback_window, forecast_steps=forecast_steps,
                device=device,
            ))
        return await run_blocking(execute, name="time_series_forecast", output_path=output_path)

    @mcp.tool()
    async def sentiment_analysis_bert(
        input_path: str | None = None, input_token: str | None = None,
        output_path: str | None = None, text_column: str | None = None,
        model_name: str = "finbert-tone", model_path: str | None = None,
        batch_size: int = 16, device: str = "auto", output_format: str = "auto",
    ) -> dict:
        if not (input_path or input_token):
            raise ValueError("input_path or input_token is required")
        def execute():
            from . import sentiment_analysis_bert as engine
            return engine.run(SimpleNamespace(
                input_path=input_path, input_token=input_token, output_path=output_path,
                text_column=text_column, model_name=model_name, model_path=model_path,
                batch_size=batch_size, device=device, output_format=output_format,
            ))
        return await run_blocking(execute, name="sentiment_analysis_bert", output_path=output_path)
