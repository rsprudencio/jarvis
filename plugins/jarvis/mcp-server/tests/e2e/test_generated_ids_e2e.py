"""Generated IDs under concurrency, against real PostgreSQL.

Observation, learning and worklog IDs are ``<namespace>::<epoch ms>``. Once
hooks and MCP tools ran on worker threads, two writes in the same millisecond
got the same ID and the insert's ON CONFLICT DO UPDATE made the second
silently replace the first while both reported "stored" (review finding F1:
23 of 120 concurrent ingests lost). IDs are now unique per process and a
generated ID never overwrites an existing row.
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import psycopg


def _rows(db_url: str, sql: str, params: tuple = ()) -> list[tuple]:
    with psycopg.connect(db_url, autocommit=True) as conn:
        return conn.execute(sql, params).fetchall()


def _parallel(n: int, fn) -> list:
    """Run ``fn(i)`` on ``n`` threads released at the same instant."""
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


def test_concurrent_generated_id_writes_all_persist(e2e_config):
    from tools.content import content_write

    n = 12
    types = ("observation", "learning", "worklog")

    def write(i: int) -> dict:
        return content_write(
            content=f"concurrent generated-id write {i}",
            content_type=types[i % len(types)],
            skip_secret_scan=True,
        )

    results = _parallel(n, write)
    assert all(r and r["success"] for r in results), results
    ids = [r["id"] for r in results]
    assert len(set(ids)) == n, f"IDs shared between writes: {ids}"

    stored = _rows(
        e2e_config["db_url"],
        "SELECT id, document FROM local.memories WHERE document LIKE 'concurrent generated-id write %%'",
    )
    assert len(stored) == n
    assert {doc for _, doc in stored} == {f"concurrent generated-id write {i}" for i in range(n)}


def test_tight_loop_writes_persist_and_list_honors_limit(e2e_config):
    """Sequential same-millisecond writes used to collapse into a few rows."""
    from tools.content import content_list, content_write

    for i in range(10):
        content_write(content=f"tight loop {i}", content_type="observation", skip_secret_scan=True)

    count = _rows(
        e2e_config["db_url"],
        "SELECT count(*) FROM local.memories WHERE document LIKE 'tight loop %%'",
    )
    assert count == [(10,)]
    listed = content_list(content_type="observation", limit=5)
    assert listed["success"]
    assert listed["returned"] == 5
    assert listed["total"] == 10


def test_generated_id_never_overwrites_an_existing_row(e2e_config, monkeypatch):
    """A taken ID (another process, a clock step back) is skipped, not replaced."""
    import tools.content as content_module

    taken_ms = 1_900_000_000_000
    first = content_module.content_write(
        content="the row that already owns this id",
        content_type="observation",
        skip_secret_scan=True,
    )
    assert first["success"]
    with psycopg.connect(e2e_config["db_url"], autocommit=True) as conn:
        conn.execute("UPDATE local.memories SET id = %s WHERE id = %s", (f"obs::{taken_ms}", first["id"]))

    # The next generated millisecond is exactly the taken one.
    monkeypatch.setattr(content_module, "_last_generated_ms", taken_ms - 1)
    monkeypatch.setattr(content_module, "time", SimpleNamespace(time=lambda: (taken_ms - 50) / 1000))

    second = content_module.content_write(
        content="a new observation in the same millisecond",
        content_type="observation",
        skip_secret_scan=True,
    )
    assert second["success"], second
    assert second["id"] == f"obs::{taken_ms + 1}"

    rows = dict(_rows(
        e2e_config["db_url"],
        "SELECT id, document FROM local.memories WHERE id IN (%s, %s)",
        (f"obs::{taken_ms}", f"obs::{taken_ms + 1}"),
    ))
    assert rows == {
        f"obs::{taken_ms}": "the row that already owns this id",
        f"obs::{taken_ms + 1}": "a new observation in the same millisecond",
    }


def test_named_ids_still_upsert(e2e_config):
    """Caller-named IDs (pattern::<name>) keep their overwrite semantics."""
    from tools.content import content_write

    r1 = content_write(content="v1", content_type="pattern", name="upsert-kept", skip_secret_scan=True)
    r2 = content_write(content="v2", content_type="pattern", name="upsert-kept", skip_secret_scan=True)
    assert r1["id"] == r2["id"]
    assert _rows(e2e_config["db_url"], "SELECT document FROM local.memories WHERE id = %s", (r1["id"],)) == [("v2",)]


def test_concurrent_hook_ingests_store_every_observation(e2e_config):
    """What the review measured: parallel Stop-hook ingests, dedup disabled."""
    from tools.hook_endpoints import ingest_auto_extract

    n = 8

    def ingest(i: int) -> dict:
        return ingest_auto_extract({
            "observations": [{
                "content": f"parallel ingest observation {i}",
                "importance_score": 0.5,
                "ingest_event_id": f"e2e-collide-{i}-obs",
            }],
            "context": {"session_id": f"s{i}"},
            "dedup": {"observation_threshold": 2.0},
        })

    results = _parallel(n, ingest)
    statuses = [r["observations"][0]["status"] for r in results]
    assert statuses == ["stored"] * n, results
    ids = {r["observations"][0]["id"] for r in results}
    assert len(ids) == n

    stored = _rows(
        e2e_config["db_url"],
        "SELECT metadata->>'ingest_event_id' FROM local.memories "
        "WHERE document LIKE 'parallel ingest observation %%'",
    )
    assert sorted(e for (e,) in stored) == sorted(f"e2e-collide-{i}-obs" for i in range(n))


def test_parallel_mcp_store_calls_store_every_learning(e2e_config):
    """Parallel jarvis_store tool calls run on the tool executor's workers."""
    import server

    n = 8
    server.shutdown_tool_executor()

    async def run() -> list:
        return await asyncio.gather(*(
            server.call_tool("jarvis_store", {
                "type": "learning",
                "content": f"parallel mcp learning {i}",
                "skip_secret_scan": True,
            })
            for i in range(n)
        ))

    try:
        responses = asyncio.run(run())
    finally:
        server.shutdown_tool_executor()
    payloads = [json.loads(r[0].text) for r in responses]
    assert all(p.get("success") for p in payloads), payloads

    stored = _rows(
        e2e_config["db_url"],
        "SELECT id FROM local.memories WHERE document LIKE 'parallel mcp learning %%'",
    )
    assert len(stored) == n


def test_overlapping_replays_of_one_ingest_event_store_it_once(e2e_config, monkeypatch):
    """The ingest_event_id check ran before the embedding, outside the INSERT
    transaction. A hook replay overlapping the write it replays (the endpoint
    answered 503 at its deadline while the worker thread went on) passed the
    check too, and both inserted: one observation stored twice."""
    import tools.document_index as document_index
    from tools.content import content_write

    n = 3
    real_prepare = document_index.prepare_document
    # Every writer has passed the early check before any of them inserts.
    all_checked = threading.Barrier(n, timeout=10)

    def prepare_after_everyone_checked(content, service):
        all_checked.wait()
        return real_prepare(content, service)

    monkeypatch.setattr(document_index, "prepare_document", prepare_after_everyone_checked)

    def replay(_i: int) -> dict:
        return content_write(
            content="observation replayed while its first write was in flight",
            content_type="observation",
            extra_metadata={"ingest_event_id": "evt-overlap-1"},
            skip_secret_scan=True,
        )

    results = _parallel(n, replay)
    assert all(r and r["success"] for r in results), results
    stored = _rows(
        e2e_config["db_url"],
        "SELECT id FROM local.memories WHERE metadata->>'ingest_event_id' = %s",
        ("evt-overlap-1",),
    )
    assert len(stored) == 1, stored
    assert {r["id"] for r in results} == {stored[0][0]}
    assert sum(1 for r in results if r.get("deduplicated")) == n - 1


def test_distinct_ingest_events_do_not_serialize_into_one(e2e_config):
    """The per-event lock is keyed on the id: different events all persist."""
    from tools.content import content_write

    n = 6

    def write(i: int) -> dict:
        return content_write(
            content=f"distinct ingest event {i}",
            content_type="observation",
            extra_metadata={"ingest_event_id": f"evt-distinct-{i}"},
            skip_secret_scan=True,
        )

    results = _parallel(n, write)
    assert all(r and r["success"] and not r.get("deduplicated") for r in results), results
    count = _rows(
        e2e_config["db_url"],
        "SELECT count(DISTINCT metadata->>'ingest_event_id') FROM local.memories"
        " WHERE metadata->>'ingest_event_id' LIKE 'evt-distinct-%%'",
    )
    assert count == [(n,)]
