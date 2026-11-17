"""Prediction backtests and Qlib benchmark operations."""

from __future__ import annotations

from types import SimpleNamespace

from ._blocking import run_blocking


def register(mcp) -> None:
    @mcp.tool()
    async def backtest_model_predictions(
        config: str, pred_path: str | None = None,
        provider_uri: str | None = None,
        start_time: str | None = None, end_time: str | None = None,
    ) -> dict:
        def execute():
            from . import backtest as engine
            return engine.run(SimpleNamespace(
                config=config, pred_path=pred_path, provider_uri=provider_uri,
                start_time=start_time, end_time=end_time,
            ))
        return await run_blocking(execute, name="backtest_model_predictions")

    @mcp.tool()
    async def qlib_benchmark_runner(
        yaml_path: str, provider_uri: str | None = None,
        experiment_name: str | None = None, task_name: str = "qlib_training",
    ) -> dict:
        def execute():
            from .training import launch_qlib_benchmark
            return launch_qlib_benchmark(
                yaml_path=yaml_path, provider_uri=provider_uri,
                experiment_name=experiment_name, task_name=task_name,
            )
        return await run_blocking(execute, name="qlib_benchmark_runner")

    @mcp.tool()
    async def qlib_benchmark_list_models() -> dict:
        def execute():
            from .training import list_qlib_models
            return list_qlib_models()
        return await run_blocking(execute, name="qlib_benchmark_list_models")
