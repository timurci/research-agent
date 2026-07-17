"""Unit tests for the MLflow eval CLI (parser and registry only)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from evals.main import MODULE_NAMES, _build_parser, main
from research_agent.shared.agent import LMConfig

if TYPE_CHECKING:
    from pathlib import Path


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
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(
        """
search-search:
  model: openai/cli-search
search-rerank:
  model: infinity/cli-rerank
""",
        encoding="utf-8",
    )
    search_cfg = LMConfig(model="openai/cli-search")
    rerank_cfg = LMConfig(model="infinity/cli-rerank")
    built: dict[str, object] = {}

    fake_module = MagicMock()
    fake_module.name = "search"
    fake_module.load_data.return_value = []
    fake_module.build_predict_fn.return_value = lambda **_: []
    fake_module.build_scorers.return_value = []

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
