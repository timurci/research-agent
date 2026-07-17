"""Unit tests for the MLflow eval CLI (parser and registry only)."""

from __future__ import annotations

import pytest

from evals.main import MODULES, _build_parser, main


def test_modules_registry_has_search_suites() -> None:
    assert set(MODULES) == {"search", "search-e2e"}
    for module in MODULES.values():
        assert module.name in MODULES
        assert callable(module.load_data)
        assert callable(module.build_predict_fn)
        assert callable(module.build_scorers)


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
        ],
    )
    assert args.modules == ["search-e2e", "search"]
    assert args.experiment == "my-exp"
    assert args.tracking_uri == "./mlruns"
    assert args.list_modules is False


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
    assert out == sorted(MODULES)


def test_main_requires_module_without_list() -> None:
    with pytest.raises(SystemExit):
        main(["--experiment", "my-exp"])


def test_main_requires_experiment_without_list() -> None:
    with pytest.raises(SystemExit):
        main(["search-e2e"])
