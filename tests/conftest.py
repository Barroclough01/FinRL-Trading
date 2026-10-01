"""Keep collection-time logging and test outputs out of runtime directories."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

_collection_workspace: TemporaryDirectory[str] | None = None
_original_cwd: Path | None = None


def pytest_sessionstart():
    # run_paper_trading opens a log file at import time, before fixtures run.
    global _collection_workspace, _original_cwd
    _original_cwd = Path.cwd()
    _collection_workspace = TemporaryDirectory(prefix="finrl-pytest-collection-")
    os.chdir(_collection_workspace.name)


def pytest_unconfigure():
    if _original_cwd is not None:
        os.chdir(_original_cwd)
    if _collection_workspace is not None:
        _collection_workspace.cleanup()


@pytest.fixture(autouse=True)
def isolated_runtime_directory(tmp_path, monkeypatch):
    """Give every test its own relative and repository-rooted output paths."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()

    import refresh_fmp_daily
    import run_paper_trading

    monkeypatch.setattr(run_paper_trading, "project_root", str(tmp_path))
    monkeypatch.setattr(refresh_fmp_daily, "SCRIPT_DIR", tmp_path)
    monkeypatch.setattr(refresh_fmp_daily, "FMP_DAILY_DIR", tmp_path / "data/fmp_daily")
