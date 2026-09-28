"""The local ONNX embedding path is safe to call from worker threads (review CR-4).

Hooks and MCP tools now encode from executor threads. Every
_ONNX_RESET_INTERVAL calls, reset_onnx_session() set _ort_session and
_tokenizer to None while other threads were between loading and running
them: AttributeError, and in ingest a dropped observation.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import tools.embedding as embedding_module
from tools.embedding import EmbeddingService

DIMS = 4


class _Output:
    def __init__(self, name):
        self.name = name


class _FakeSession:
    def run(self, _names, inputs):
        time.sleep(0.0005)  # widen the window a concurrent reset could hit
        batch = inputs["input_ids"].shape[0]
        return [np.ones((batch, DIMS), dtype=np.float32)]

    def get_outputs(self):
        return [_Output("sentence_embedding")]


def _fake_tokenizer(texts, **kwargs):
    n = len(texts)
    return {"input_ids": np.ones((n, 3), dtype=np.int64), "attention_mask": np.ones((n, 3), dtype=np.int64)}


@pytest.fixture
def onnx_service(monkeypatch):
    svc = EmbeddingService(model_name="fake-model", dimensions=DIMS, backend="onnx")

    def fake_load(self):
        if self._ort_session is None:
            time.sleep(0.0005)
            self._ort_session = _FakeSession()
            self._tokenizer = _fake_tokenizer
            self._onnx_call_count = 0

    monkeypatch.setattr(EmbeddingService, "_load_onnx_locked", fake_load)
    monkeypatch.setattr(EmbeddingService, "_ONNX_RESET_INTERVAL", 5)
    monkeypatch.setattr("gc.collect", lambda *args: 0)
    return svc


def test_concurrent_encodes_survive_periodic_session_resets(onnx_service):
    errors: list[str] = []
    ok = []
    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        for i in range(60):
            try:
                vec = onnx_service.encode(f"text {i}")
                assert len(vec) == DIMS
                ok.append(1)
            except Exception as exc:  # pragma: no cover - the regression
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    assert len(ok) == 240


def test_concurrent_batch_encodes_survive_resets(onnx_service):
    errors: list[str] = []

    def worker():
        for _ in range(20):
            try:
                assert len(onnx_service.encode_batch(["a", "b", "c"], batch_size=2)) == 3
            except Exception as exc:  # pragma: no cover - the regression
                errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []


def test_singleton_is_created_once_under_concurrency(monkeypatch):
    import tools.config as config

    monkeypatch.setattr(config, "get_embedding_config", lambda: {
        "model": "fake-model", "dimensions": DIMS, "device": "cpu", "backend": "onnx",
    })
    monkeypatch.setattr(embedding_module, "_effective_model_name", lambda cfg: "fake-model")
    monkeypatch.setattr(embedding_module, "_service", None)
    monkeypatch.setattr(embedding_module, "_service_cache_key", None)
    created = []
    real_init = EmbeddingService.__init__

    def slow_init(self, *args, **kwargs):
        time.sleep(0.05)  # widen the check-then-create window
        created.append(self)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(EmbeddingService, "__init__", slow_init)
    services = []
    barrier = threading.Barrier(6)

    def worker():
        barrier.wait()
        services.append(embedding_module.get_embedding_service())

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert len(created) == 1
    assert len({id(s) for s in services}) == 1
