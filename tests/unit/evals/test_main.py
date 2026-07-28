"""Unit tests for the Opik eval CLI (parser and registry only)."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.harness import EVAL_SEED, EvalModule
from evals.main import MODULE_NAMES, _build_parser, main
from research_agent.shared.config.instructions import (
    DEFAULT_INSTRUCTIONS_CONFIG_PATH,
)
from research_agent.shared.config.models import LMConfig

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
    rows = [{"query": f"query number {i}"} for i in range(row_count)]
    return EvalModule(
        name=name,
        load_data=lambda: rows,
        build_task=lambda: lambda _item: {"papers": []},
        build_scorers=list,
        sample_limit=limit,
    )


def _mock_opik_evaluate(opik_mod: MagicMock) -> None:
    mock_result = MagicMock()
    mock_result.experiment_name = "cli-exp"
    mock_result.experiment_url = "http://localhost/test"
    opik_mod.evaluate.return_value = mock_result
    opik_mod.Opik.return_value.get_or_create_dataset.return_value = MagicMock()


def test_module_names_has_search_suite() -> None:
    assert frozenset({"search-search"}) == MODULE_NAMES


def test_parser_accepts_modules_and_options() -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "search-search",
            "--experiment",
            "my-exp",
            "--config",
            "config/custom-lm.yaml",
            "--instructions",
            "config/custom-instructions.yaml",
        ],
    )
    assert args.modules == ["search-search"]
    assert args.experiment == "my-exp"
    assert args.config.as_posix() == "config/custom-lm.yaml"
    assert args.instructions.as_posix() == "config/custom-instructions.yaml"
    assert args.list_modules is False


def test_parser_config_defaults_to_lm_yaml() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--list"])
    assert args.config.as_posix() == "config/lm.yaml"


def test_parser_instructions_defaults_to_instructions_yaml() -> None:
    parser = _build_parser()
    args = parser.parse_args(["--list"])
    assert args.instructions == DEFAULT_INSTRUCTIONS_CONFIG_PATH


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
        ["search-search", "--experiment", "my-exp", "--limit", "5", "--seed", "7"],
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
        main(["search-search"])


def test_main_loads_config_and_injects_into_build_modules(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    instructions_path = tmp_path / "instructions.yaml"
    instructions_path.write_text(
        "instructions:\n  search-search: data/optimize/output/search-search.json\n",
        encoding="utf-8",
    )
    search_cfg = LMConfig(model="openai/cli-search")
    rerank_cfg = LMConfig(model="infinity/cli-rerank")
    built: dict[str, object] = {}

    fake_module = MagicMock()
    fake_module.name = "search-search"
    fake_module.load_data.return_value = []
    fake_module.build_task.return_value = lambda _item: {"papers": []}
    fake_module.build_scorers.return_value = []
    fake_module.sample_limit = None

    def _capture_build(
        *,
        search_lm_config: LMConfig,
        rerank_lm_config: LMConfig,
        instructions: dict[str, object] | None,
    ) -> dict[str, object]:
        built["search-search"] = search_lm_config
        built["rerank"] = rerank_lm_config
        built["instructions"] = instructions
        return {"search-search": fake_module}

    with (
        patch("evals.main.build_modules", side_effect=_capture_build),
        patch("evals.main.opik") as opik_mod,
        patch("evals.main.dspy"),
    ):
        _mock_opik_evaluate(opik_mod)
        main(
            [
                "search-search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--instructions",
                str(instructions_path),
            ],
        )

    assert built["search-search"] == search_cfg
    assert built["rerank"] == rerank_cfg
    assert built["instructions"] == {
        "search-search": Path("data/optimize/output/search-search.json"),
    }


def test_main_subsamples_rows_over_module_limit(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search-search", row_count=10, limit=3)

    with (
        patch("evals.main.build_modules", return_value={"search-search": module}),
        patch("evals.main.opik") as opik_mod,
        patch("evals.main.dspy"),
        patch("evals.main.load_instructions_config", return_value={}),
        caplog.at_level(logging.INFO, logger="evals.main"),
    ):
        _mock_opik_evaluate(opik_mod)
        main(["search-search", "--experiment", "cli-exp", "--config", str(config_path)])

    assert "loaded 10 rows" in caplog.text
    assert "exceed sample limit 3" in caplog.text


def test_main_limit_flag_overrides_module_sample_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search-search", row_count=10, limit=50)

    with (
        patch("evals.main.build_modules", return_value={"search-search": module}),
        patch("evals.main.opik") as opik_mod,
        patch("evals.main.dspy"),
        patch("evals.main.load_instructions_config", return_value={}),
    ):
        _mock_opik_evaluate(opik_mod)
        main(
            [
                "search-search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--limit",
                "2",
            ],
        )


def test_main_rejects_zero_limit(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "search-search",
                "--experiment",
                "cli-exp",
                "--config",
                str(config_path),
                "--limit",
                "0",
            ],
        )


def test_main_configures_dspy_opik_callback(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search-search", row_count=1, limit=1)

    with (
        patch("evals.main.dspy") as fake_dspy,
        patch("evals.main.build_modules", return_value={"search-search": module}),
        patch("evals.main.opik") as opik_mod,
    ):
        _mock_opik_evaluate(opik_mod)
        main(["search-search", "--experiment", "cli-exp", "--config", str(config_path)])

    fake_dspy.configure.assert_called_once()
    callback_arg = fake_dspy.configure.call_args.kwargs["callbacks"]
    assert len(callback_arg) == 1


def test_main_disables_dspy_disk_cache(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _rows_module("search-search", row_count=1, limit=1)

    with (
        patch("evals.main.dspy") as fake_dspy,
        patch("evals.main.build_modules", return_value={"search-search": module}),
        patch("evals.main.opik") as opik_mod,
    ):
        _mock_opik_evaluate(opik_mod)
        main(["search-search", "--experiment", "cli-exp", "--config", str(config_path)])

    fake_dspy.configure_cache.assert_called_once_with(
        enable_disk_cache=False,
        enable_memory_cache=True,
    )


def test_main_list_does_not_disable_dspy_disk_cache(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with patch("evals.main.dspy") as fake_dspy:
        main(["--list"])

    out = capsys.readouterr().out.strip().splitlines()
    assert out == sorted(MODULE_NAMES)
    fake_dspy.configure_cache.assert_not_called()
