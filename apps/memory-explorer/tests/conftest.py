"""Shared fixtures for the memory-explorer tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as app_module  # noqa: E402


@pytest.fixture(autouse=True)
def _no_stale_unreachable_verdict(monkeypatch):
    """A probe that failed in one test (closed port, fake recovering server)
    would otherwise make local-DB requests in the next tests fail fast."""
    monkeypatch.setattr(app_module, "_db_unreachable_at", None)
