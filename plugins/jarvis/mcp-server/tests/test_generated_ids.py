"""Clock-generated memory IDs are unique and never overwrite (review F1).

obs::/learning::/worklog:: IDs are epoch milliseconds. With hooks and MCP
tools on worker threads, same-millisecond writes shared an ID and the upsert
silently replaced the earlier row. The real-PostgreSQL side (DO NOTHING +
regenerate, concurrent ingests and tool calls) is in
tests/e2e/test_generated_ids_e2e.py.
"""

from __future__ import annotations

import contextlib
import threading

import tools.content as content_module
from tools.content import content_write


def _parallel(n: int, fn) -> list:
    barrier = threading.Barrier(n)
    results: list = [None] * n

    def worker(i: int) -> None:
        barrier.wait()
        results[i] = fn(i)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    return results


def test_unique_id_ms_is_strictly_increasing_across_threads():
    per_thread = 500
    results = _parallel(8, lambda i: [content_module._unique_id_ms() for _ in range(per_thread)])
    values = [v for chunk in results for v in chunk]
    assert len(set(values)) == len(values) == 8 * per_thread
    for chunk in results:
        assert chunk == sorted(chunk)


def test_concurrent_generated_writes_get_distinct_ids(mock_config):
    types = ("observation", "learning", "worklog")
    results = _parallel(24, lambda i: content_write(
        content=f"parallel write {i}", content_type=types[i % 3], skip_secret_scan=True,
    ))
    assert all(r["success"] for r in results)
    ids = [r["id"] for r in results]
    assert len(set(ids)) == 24
    stored = {row["id"] for row in mock_config.db.core_rows.values()}
    assert set(ids) <= stored


class _RecordingCursor:
    """Cursor whose first memory INSERT hits an existing row (rowcount 0)."""

    def __init__(self, log: list, conflicts: int):
        self.log = log
        self.conflicts = conflicts
        self.rowcount = 1

    def execute(self, sql, params=None):
        is_insert = sql.lstrip().upper().startswith("INSERT INTO LOCAL.MEMORIES")
        if is_insert:
            self.log.append((sql, params[0]))
            if self.conflicts > 0:
                self.conflicts -= 1
                self.rowcount = 0
                return
        self.rowcount = 1

    def executemany(self, sql, params_list):
        self.rowcount = len(params_list)

    def fetchone(self):
        return None

    def fetchall(self):
        return []


class _RecordingPool:
    def __init__(self, log: list, conflicts: int):
        self.cursor_obj = _RecordingCursor(log, conflicts)

    @contextlib.contextmanager
    def connection(self, timeout=None):
        pool = self

        class Conn:
            @contextlib.contextmanager
            def cursor(self):
                yield pool.cursor_obj

            def commit(self):
                pass

        yield Conn()


def _write_with_conflicts(monkeypatch, content_type: str, conflicts: int, **kwargs):
    import tools.schema as schema

    log: list = []
    monkeypatch.setattr(schema, "_get_pool", lambda: _RecordingPool(log, conflicts))
    result = content_write(content="x", content_type=content_type, skip_secret_scan=True, **kwargs)
    return result, log


def test_taken_generated_id_is_skipped_not_overwritten(mock_config, monkeypatch):
    result, log = _write_with_conflicts(monkeypatch, "observation", conflicts=2)
    assert result["success"], result
    assert len(log) == 3
    assert all("ON CONFLICT (id) DO NOTHING" in sql for sql, _ in log)
    first, second, third = (doc_id for _, doc_id in log)
    assert len({first, second, third}) == 3
    assert result["id"] == third


def test_generated_id_gives_up_after_bounded_attempts(mock_config, monkeypatch):
    result, log = _write_with_conflicts(
        monkeypatch, "worklog", conflicts=content_module._GENERATED_ID_ATTEMPTS
    )
    assert result["success"] is False
    assert "no free worklog:: id" in result["error"]
    assert len(log) == content_module._GENERATED_ID_ATTEMPTS


def test_named_ids_keep_upsert_semantics(mock_config, monkeypatch):
    result, log = _write_with_conflicts(monkeypatch, "pattern", conflicts=0, name="kept")
    assert result["success"]
    assert len(log) == 1
    assert "ON CONFLICT (id) DO UPDATE" in log[0][0]
    assert result["id"] == "pattern::kept"
