"""Unit tests for the MLflow eval CLI (parser and registry only)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from evals.harness import EVAL_SEED, EvalModule, sample_rows
from evals.main import MODULE_NAMES, _build_parser, main
from research_agent.shared.agent import LMConfig

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG_YAML = """
search-search:
  model: openai/cli-search
search-rerank:
  model: infinity/cli-rerank
"""


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(_CONFIG_YAML, encoding="utf-8")
    return config_path


def _rows_module(name: str, row_count: int, limit: int | None) -> EvalModule:
    rows = [{"inputs": {"query": f"query number {i}"}} for i in range(row_count)]
    return EvalModule(
        name=name,
        load_data=lambda: rows,
        build_predict_fn=lambda: lambda **_: [],
        build_scorers=list,
        sample_limit=limit,
    )


def _mock_mlflow_evaluate(mlflow_mod: MagicMock) -> None:
    mlflow_mod.start_run.return_value.__enter__.return_value = None
    mlflow_mod.start_run.return_value.__exit__.return_value = None
    mlflow_mod.genai.evaluate.return_value = MagicMock(passed=True, reason="ok")


def test_module_names_has_search_suites() -> None:
    assert frozenset({"search", "search-e2e"}) == MODULE_NAMES


def test_parser_accepts_modules_and_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search-e2e",
            "search",
            "--experiment",
            "my-exp",
            "--tracking-uri",
            "./mlruns",
            "--config",
            "config/custom-lm.yaml",
        ],
    )
    assert args.modules == ["search-e2e", "search"]
    assert args.experiment == "my-exp"
    assert args.tracking_uri == "./mlruns"
    assert args.config.as_posix() == "config/custom-lm.yaml"
    assert args.list_modules is False


def test_parser_config_defaults_to_lm_yaml() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--list"])
    assert args.config.as_posix() == "config/lm.yaml"


def test_parser_list_flag() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--list"])
    assert args.list_modules is True
    assert args.modules == []
    assert args.experiment is None


def test_parser_limit_and_seed_defaults() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--list"])
    assert args.limit is None
    assert args.seed == EVAL_SEED


def test_parser_accepts_limit_and_seed() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        ["search", "--experiment", "my-exp", "--limit", "5", "--seed", "7"],
    )
    assert args.limit == 5
    assert args.seed == 7


def test_parser_rejects_unknown_module() -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--experiment", "my-exp", "not-a-module"])


def test_main_list_prints_modules(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--list"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == sorted(MODULE_NAMES)


def test_main_requires_module_without_list() -> None:
    with pytest.raises(SystemExit):
        main(["--experiment", "my-exp"])


def test_main_requires_experiment_without_list() -> None:
    with pytest.raises(SystemExit):
        main(["search-e2e"])


def test_main_loads_config_and_injects_into_build_modules(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    search_cfg = LMConfig(model="openai/cli-search")
    rerank_cfg = LMConfig(model="infinity/cli-rerank")
    built: dict[str, object] = {}

    fake_module = MagicMock()
    fake_module.name = "search"
    fake_module.load_data.return_value = []
    fake_module.build_predict_fn.return_value = lambda **_: []
    fake_module.build_scorers.return_value = []
    fake_module.sample_limit = None

    def _capture_build(
        *,
        search_lm_config: LMConfig,
        rerank_lm_config: LMConfig,
    ) -> dict[str, object]:
        built["search"] = search_lm_config
        built["rerank"] = rerank_lm_config
        return {"search": fake_module}

    with (
        patch("evals.main.build_modules", side_effect=_capture_build),
        patch("evals.main.mlflow") as mlflow_mod,
    ):
        mlflow_mod.start_run.return_value.__enter__.return_value = None
        mlflow_mod.start_run.return_value.__exit__.return_value = None
        mlflow_mod.genai.evaluate.return_value = MagicMock(
            passed=True,
            reason="ok",
        )
        main(
            [
                "search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
            ],
        )

    assert built["search"] == search_cfg
    assert built["rerank"] == rerank_cfg


def test_main_subsamples_rows_over_module_limit(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search", row_count=10, limit=3)

    with (
        patch("evals.main.build_modules", return_value={"search": module}),
        patch("evals.main.mlflow") as mlflow_mod,
        caplog.at_level(logging.INFO, logger="evals.main"),
    ):
        _mock_mlflow_evaluate(mlflow_mod)
        main(["search", "--experiment", "cli-exp", "--config", str(config_path)])

    evaluate_data = mlflow_mod.genai.evaluate.call_args.kwargs["data"]
    assert len(evaluate_data) == 3
    assert evaluate_data == sample_rows(module.load_data(), limit=3, seed=EVAL_SEED)
    mlflow_mod.log_params.assert_called_once_with(
        {"eval.rows": 3, "eval.seed": EVAL_SEED, "eval.sample_limit": 3},
    )
    assert "loaded 10 rows" in caplog.text
    assert "exceed sample limit 3" in caplog.text


def test_main_limit_flag_overrides_module_sample_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search", row_count=10, limit=50)

    with (
        patch("evals.main.build_modules", return_value={"search": module}),
        patch("evals.main.mlflow") as mlflow_mod,
    ):
        _mock_mlflow_evaluate(mlflow_mod)
        main(
            [
                "search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--limit",
                "2",
            ],
        )

    evaluate_data = mlflow_mod.genai.evaluate.call_args.kwargs["data"]
    assert len(evaluate_data) == 2


def test_main_seed_flag_flows_to_sampling(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search", row_count=10, limit=5)

    with (
        patch("evals.main.build_modules", return_value={"search": module}),
        patch("evals.main.mlflow") as mlflow_mod,
    ):
        _mock_mlflow_evaluate(mlflow_mod)
        main(
            [
                "search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--seed",
                "7",
            ],
        )

    evaluate_data = mlflow_mod.genai.evaluate.call_args.kwargs["data"]
    assert evaluate_data == sample_rows(module.load_data(), limit=5, seed=7)
    mlflow_mod.log_params.assert_called_once_with(
        {"eval.rows": 5, "eval.seed": 7, "eval.sample_limit": 5},
    )


def test_main_rejects_zero_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--limit",
                "0",
            ],
        )
