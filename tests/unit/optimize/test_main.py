"""Unit tests for optimize CLI entrypoint."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import dspy
import pytest

from optimize.main import _build_parser, main
from optimize.search.modules import OptimizeModule
from research_agent.search.models import ResearchQuery

if TYPE_CHECKING:
    from pathlib import Path

_CONFIG_YAML = """
search-search:
  model: openai/cli-search
search-rerank:
  model: infinity/cli-rerank
gepa-reflection:
  model: openai/cli-teacher
"""


def _write_config(tmp_path: Path) -> Path:
    config_path = tmp_path / "lm.yaml"
    config_path.write_text(_CONFIG_YAML, encoding="utf-8")
    return config_path


def _fake_module() -> OptimizeModule:
    example = dspy.Example(
        research_query=ResearchQuery(text="quantum error correction codes"),
    ).with_inputs("research_query")
    return OptimizeModule(
        name="search-search",
        load_trainset=lambda: [example],
        metric=MagicMock(name="metric"),
        build_student=lambda: MagicMock(name="student"),
        sample_limit=None,
    )


def test_main_list_prints_modules(capsys: pytest.CaptureFixture[str]) -> None:
    main(["--list"])
    out = capsys.readouterr().out.strip().splitlines()
    assert out == ["search-search"]


def test_main_requires_modules() -> None:
    with pytest.raises(SystemExit):
        main([])


def test_parser_budget_defaults_to_light() -> None:
    args = _build_parser().parse_args(["--list"])
    assert args.budget == "light"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("light", "light"),
        ("medium", "medium"),
        ("heavy", "heavy"),
        ("1", 1),
        ("20", 20),
    ],
)
def test_parser_budget_accepts_presets_and_positive_ints(
    value: str,
    expected: str | int,
) -> None:
    args = _build_parser().parse_args(["--budget", value, "--list"])
    assert args.budget == expected


@pytest.mark.parametrize("value", ["0", "-1", "turbo", "1.5"])
def test_parser_budget_rejects_invalid_values(value: str) -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--budget", value, "--list"])


def test_main_runs_gepa_and_saves_program(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _fake_module()
    optimized = MagicMock()
    optimized.detailed_results.highest_score_achieved_per_val_task = [0.5]
    gepa = MagicMock()
    gepa.compile.return_value = optimized

    with (
        patch("optimize.main.build_modules", return_value={"search-search": module}),
        patch("optimize.main.dspy.GEPA", return_value=gepa) as gepa_cls,
    ):
        main(
            [
                "search-search",
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path),
            ],
        )

    gepa_cls.assert_called_once()
    assert gepa_cls.call_args.kwargs["auto"] == "light"
    assert gepa_cls.call_args.kwargs["max_full_evals"] is None
    assert gepa_cls.call_args.kwargs["track_stats"] is True
    assert gepa_cls.call_args.kwargs["log_dir"] == str(tmp_path / "search-search")
    assert gepa_cls.call_args.kwargs["seed"] == 0
    gepa.compile.assert_called_once()
    examples = module.load_trainset()
    assert gepa.compile.call_args.kwargs["trainset"] == examples
    assert gepa.compile.call_args.kwargs["valset"] == examples
    optimized.save.assert_called_once_with(str(tmp_path / "search-search.json"))


def test_main_budget_flag_selects_gepa_preset(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _fake_module()
    optimized = MagicMock()
    optimized.detailed_results = None
    gepa = MagicMock()
    gepa.compile.return_value = optimized

    with (
        patch("optimize.main.build_modules", return_value={"search-search": module}),
        patch("optimize.main.dspy.GEPA", return_value=gepa) as gepa_cls,
    ):
        main(
            [
                "search-search",
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path),
                "--budget",
                "heavy",
            ],
        )

    assert gepa_cls.call_args.kwargs["auto"] == "heavy"
    assert gepa_cls.call_args.kwargs["max_full_evals"] is None


def test_main_budget_integer_sets_max_full_evals(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    module = _fake_module()
    optimized = MagicMock()
    optimized.detailed_results = None
    gepa = MagicMock()
    gepa.compile.return_value = optimized

    with (
        patch("optimize.main.build_modules", return_value={"search-search": module}),
        patch("optimize.main.dspy.GEPA", return_value=gepa) as gepa_cls,
    ):
        main(
            [
                "search-search",
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path),
                "--budget",
                "20",
            ],
        )

    assert gepa_cls.call_args.kwargs["auto"] is None
    assert gepa_cls.call_args.kwargs["max_full_evals"] == 20


def _module_with_pool_size(pool_size: int) -> OptimizeModule:
    examples = [
        dspy.Example(
            research_query=ResearchQuery(text=f"query number {i}")
        ).with_inputs("research_query")
        for i in range(pool_size)
    ]
    return OptimizeModule(
        name="search-search",
        load_trainset=lambda: examples,
        metric=MagicMock(name="metric"),
        build_student=lambda: MagicMock(name="student"),
        sample_limit=None,
    )


def test_main_applies_low_pool_split_when_pool_below_threshold(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    module = _module_with_pool_size(50)
    optimized = MagicMock()
    optimized.detailed_results = None
    gepa = MagicMock()
    gepa.compile.return_value = optimized

    with (
        patch("optimize.main.build_modules", return_value={"search-search": module}),
        patch("optimize.main.dspy.GEPA", return_value=gepa),
    ):
        main(
            [
                "search-search",
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path),
            ],
        )

    trainset = gepa.compile.call_args.kwargs["trainset"]
    valset = gepa.compile.call_args.kwargs["valset"]
    assert len(trainset) == 25
    assert len(valset) == 25


def test_main_applies_default_split_when_pool_at_or_above_threshold(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path)
    module = _module_with_pool_size(200)
    optimized = MagicMock()
    optimized.detailed_results = None
    gepa = MagicMock()
    gepa.compile.return_value = optimized

    with (
        patch("optimize.main.build_modules", return_value={"search-search": module}),
        patch("optimize.main.dspy.GEPA", return_value=gepa),
    ):
        main(
            [
                "search-search",
                "--config",
                str(config_path),
                "--out-dir",
                str(tmp_path),
            ],
        )

    trainset = gepa.compile.call_args.kwargs["trainset"]
    valset = gepa.compile.call_args.kwargs["valset"]
    assert len(trainset) == 160
    assert len(valset) == 40
