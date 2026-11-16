"""Local model training and background research-task tools.

``train`` runs a local sklearn/Qlib job and blocks until it finishes.  The
remaining trainers launch external research-project scripts in the background
through :mod:`financial_tool.tools.training`, which owns the task registry.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from ._blocking import run_blocking


def register(mcp) -> None:
    @mcp.tool()
    async def train(config: str, model_path: str) -> dict:
        def execute():
            from . import train as engine
            return engine.run(SimpleNamespace(config=config, model_path=model_path))
        return await run_blocking(execute, name="train", output_path=model_path)

    @mcp.tool()
    async def train_qcm(
        model: str = "qrdqn", seed: int = 0, pool: int = 20,
        std_lam: float = 1.0, task_name: str = "training",
    ) -> dict:
        def execute():
            from .training import launch_training
            return launch_training(
                "train_qcm", task_name=task_name,
                model=model, seed=seed, pool=pool, std_lam=std_lam,
            )
        return await run_blocking(execute, name="train_qcm")

    @mcp.tool()
    async def train_AFF(
        config_file: str | None = None, config_params: dict[str, Any] | None = None,
        task_name: str = "training",
    ) -> dict:
        def execute():
            from .training import launch_training
            return launch_training(
                "train_AFF", config_file=config_file,
                config_params=config_params, task_name=task_name,
            )
        return await run_blocking(execute, name="train_AFF")

    @mcp.tool()
    async def train_gfn_AlphaSAGE(
        config_file: str | None = None, config_params: dict[str, Any] | None = None,
        task_name: str = "training",
    ) -> dict:
        def execute():
            from .training import launch_training
            return launch_training(
                "train_gfn_AlphaSAGE", config_file=config_file,
                config_params=config_params, task_name=task_name,
            )
        return await run_blocking(execute, name="train_gfn_AlphaSAGE")

    @mcp.tool()
    async def train_GP_AlphaSAGE(
        config_file: str | None = None, config_params: dict[str, Any] | None = None,
        task_name: str = "training",
    ) -> dict:
        def execute():
            from .training import launch_training
            return launch_training(
                "train_GP_AlphaSAGE", config_file=config_file,
                config_params=config_params, task_name=task_name,
            )
        return await run_blocking(execute, name="train_GP_AlphaSAGE")

    @mcp.tool()
    async def train_PPO_AlphaSAGE(
        config_file: str | None = None, config_params: dict[str, Any] | None = None,
        task_name: str = "training",
    ) -> dict:
        def execute():
            from .training import launch_training
            return launch_training(
                "train_PPO_AlphaSAGE", config_file=config_file,
                config_params=config_params, task_name=task_name,
            )
        return await run_blocking(execute, name="train_PPO_AlphaSAGE")

    @mcp.tool()
    async def training_get_task(
        task_id: str, tail: int = 50,
    ) -> dict:
        def execute():
            from .training import get_training_status
            return get_training_status(task_id, tail=tail)
        return await run_blocking(execute, name="training_get_task")

    @mcp.tool()
    async def training_list_tasks() -> dict:
        def execute():
            from .training import list_training_tasks
            return {"tasks": list_training_tasks()}
        return await run_blocking(execute, name="training_list_tasks")