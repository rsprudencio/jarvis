"""Outage resilience against real PostgreSQL (2026-09 disk-full incident).

The pool talks to the e2e database through a local TCP front that can switch,
mid-test, into a server rejecting every login with 57P03 "recovery mode" —
what the embedded PostgreSQL did for ~15h. The same pool object must fail fast
while it is "down" and recover on its own once it is "back".

Also verifies the idx_retrieval_candidates_event drop: the migration is
idempotent and the primary key really serves every access path it covered.
"""

from __future__ import annotations

import socket
import threading
import time
from urllib.parse import urlparse, urlunparse

import psycopg
import pytest
from psycopg_pool.base import AttemptWithBackoff

from tests.fake_pg_server import FakeRecoveringPostgres, reject_login_in_recovery


class SwitchablePostgres:
    """127.0.0.1 front for the e2e PostgreSQL that can play a crash-looping one.

    ``forward`` pipes bytes to the real server. ``crash()`` drops every live
    connection (like a PANIC restart) and answers new logins with 57P03 until
    ``recover()``.
    """

    def __init__(self, upstream_host: str, upstream_port: int) -> None:
        self._upstream = (upstream_host, upstream_port)
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(64)
        self._sock.settimeout(0.2)
        self.port = self._sock.getsockname()[1]
        self.recovering = False
        self._live: set[socket.socket] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def crash(self) -> None:
        self.recovering = True
        with self._lock:
            live, self._live = self._live, set()
        for sock in live:
            _hard_close(sock)

    def recover(self) -> None:
        self.recovering = False

    def close(self) -> None:
        self._stop.set()
        self.crash()
        self._thread.join(timeout=2)
        self._sock.close()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            if self.recovering:
                threading.Thread(
                    target=reject_login_in_recovery, args=(client,), daemon=True
                ).start()
                continue
            try:
                upstream = socket.create_connection(self._upstream, timeout=3)
            except OSError:
                client.close()
                continue
            upstream.settimeout(None)
            with self._lock:
                self._live.update((client, upstream))
            for src, dst in ((client, upstream), (upstream, client)):
                threading.Thread(target=self._pump, args=(src, dst), daemon=True).start()

    @staticmethod
    def _pump(src: socket.socket, dst: socket.socket) -> None:
        try:
            while data := src.recv(65536):
                dst.sendall(data)
        except OSError:
            pass
        finally:
            _hard_close(src)
            _hard_close(dst)


def _hard_close(sock: socket.socket) -> None:
    try:
        sock.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    sock.close()


@pytest.fixture
def switchable_db(e2e_config, monkeypatch):
    """Point the production pool at a SwitchablePostgres front, fast timings."""
    import tools.schema as schema

    parsed = urlparse(e2e_config["db_url"])
    front = SwitchablePostgres(parsed.hostname, parsed.port or 5432)
    credentials = parsed.netloc.rsplit("@", 1)[0] + "@" if "@" in parsed.netloc else ""
    front_url = urlunparse(parsed._replace(netloc=f"{credentials}127.0.0.1:{front.port}"))

    # Test-only timing: a 0.5s breaker window and near-immediate pool
    # reconnects keep the down→up cycle inside a couple of seconds.
    monkeypatch.setattr(schema, "_BREAKER_OPEN_SECONDS", 0.5)
    monkeypatch.setattr(AttemptWithBackoff, "INITIAL_DELAY", 0.2)
    monkeypatch.setattr(AttemptWithBackoff, "DELAY_BACKOFF", 1.0)
    monkeypatch.setenv("POSTGRES_URL", front_url)
    schema.reset_pool()
    try:
        yield front
    finally:
        schema.reset_pool()
        front.close()


def _observation_payload(tag: str) -> dict:
    return {
        "observations": [{
            "content": f"Outage drill observation {tag}: the pool must fail fast",
            "importance_score": 0.6,
            "ingest_event_id": f"outage-e2e:{tag}",
        }],
        "worklog": {
            "task_summary": f"Outage drill worklog {tag}",
            "workstream": "Jarvis Plugin",
            "ingest_event_id": f"outage-e2e:wl:{tag}",
        },
        "context": {"session_id": "outage-e2e"},
        "dedup": {"observation_threshold": 0.95, "worklog_threshold": 0.9},
    }


def _wait_until_available(timeout: float = 6.0) -> float:
    """Poll a hook-sized query until the breaker closes; return seconds taken."""
    from tools.schema import DatabaseUnavailable, execute_query

    started = time.monotonic()
    while time.monotonic() - started < timeout:
        try:
            row = execute_query("SELECT 1 AS ok", fetch="one", conn_timeout=1.5)
        except DatabaseUnavailable:
            time.sleep(0.1)
            continue
        assert row == {"ok": 1}
        return time.monotonic() - started
    raise AssertionError(f"database did not come back within {timeout}s")


def test_breaker_fails_fast_during_outage_and_recovers(
    switchable_db, e2e_config, tmp_path, monkeypatch
):
    from tools import schema
    from tools.hook_endpoints import get_prompt_context, ingest_auto_extract

    # ── Healthy: real writes through the resilient pool ──────────────
    monkeypatch.setenv("PGDATA", str(tmp_path))
    status = schema.probe_db_status()
    assert status["status"] == "ok", status
    assert isinstance(status["free_bytes"], int)

    healthy = ingest_auto_extract(_observation_payload("before"))
    assert healthy["success"] is True, healthy
    assert healthy["observations"][0]["status"] == "stored"
    assert healthy["worklog"]["status"] == "stored"
    assert schema.db_available() is True

    # ── Outage: the database crash-loops in recovery mode ────────────
    switchable_db.crash()

    started = time.monotonic()
    down = ingest_auto_extract(_observation_payload("during"))
    first_failure = time.monotonic() - started
    assert first_failure < 2.5, f"hook ingest took {first_failure:.2f}s"
    assert down["success"] is False
    assert down["retryable"] is True
    assert down["error"].startswith("database unavailable: ")
    assert "recovery mode" in down["error"]
    assert "jarvis:jarvis" not in down["error"]
    assert schema.db_available() is False

    # Breaker open: every hook path answers without touching the pool.
    started = time.monotonic()
    again = ingest_auto_extract(_observation_payload("during-2"))
    context = get_prompt_context("what did we decide about the outage drill?")
    with pytest.raises(schema.DatabaseUnavailable):
        with schema.checkout_timeout():
            schema.execute_query("SELECT 1", fetch="one")
    assert time.monotonic() - started < 0.5
    assert again["retryable"] is True
    assert context["degraded"] is True and context["matches"] == []

    probe = schema.probe_db_status()
    assert probe["status"] == "recovering", probe
    assert "recovery mode" in probe["error"]
    assert schema.get_db_status()["status"] == "recovering"

    # Nothing from the outage window was written.
    with psycopg.connect(e2e_config["db_url"]) as conn:
        leaked = conn.execute(
            "SELECT count(*) FROM local.memories "
            "WHERE metadata->>'ingest_event_id' LIKE %s",
            ("outage-e2e:%during%",),
        ).fetchone()[0]
    assert leaked == 0

    # ── Recovery: same pool object, breaker closes by itself ─────────
    switchable_db.recover()
    took = _wait_until_available()
    assert took < 5.0
    assert schema.db_available() is True

    replay = ingest_auto_extract(_observation_payload("during"))
    assert replay["success"] is True, replay
    assert replay["observations"][0]["status"] == "stored"
    assert schema.probe_db_status()["status"] == "ok"


def test_hook_call_fails_fast_against_recovering_server(e2e_config, monkeypatch):
    """A pool that never reached the database: fail under 2.5s, no password leak."""
    from tools import schema
    from tools.hook_endpoints import ingest_auto_extract

    with FakeRecoveringPostgres() as fake:
        monkeypatch.setenv("POSTGRES_URL", fake.url())
        schema.reset_pool()
        try:
            started = time.monotonic()
            result = ingest_auto_extract(_observation_payload("fake"))
            elapsed = time.monotonic() - started

            assert elapsed < 2.5, f"took {elapsed:.2f}s"
            assert result["success"] is False and result["retryable"] is True
            assert "recovery mode" in result["error"]
            assert "hunter2-secret" not in result["error"]

            probe = schema.probe_db_status()
            assert probe["status"] == "recovering"
            assert "hunter2-secret" not in probe["error"]
        finally:
            schema.reset_pool()

    # Pointed back at the real database, the fresh pool works at once.
    monkeypatch.setenv("POSTGRES_URL", e2e_config["db_url"])
    assert schema.probe_db_status()["status"] == "ok"
    assert schema.db_available() is True
    assert schema.execute_query("SELECT 1 AS ok", fetch="one") == {"ok": 1}


# ── idx_retrieval_candidates_event drop ──────────────────────────────


def _index_names(conn) -> set[str]:
    rows = conn.execute(
        "SELECT indexname FROM pg_indexes "
        "WHERE schemaname = 'local' AND tablename = 'retrieval_candidates'"
    ).fetchall()
    return {row[0] for row in rows}


def test_redundant_candidate_index_drop_is_idempotent(e2e_config):
    from tools.schema import ensure_schema, reset_pool

    with psycopg.connect(e2e_config["db_url"], autocommit=True) as conn:
        # An install from before the drop still has the index.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_candidates_event "
            "ON local.retrieval_candidates (event_id, vector_rank)"
        )
        assert "idx_retrieval_candidates_event" in _index_names(conn)

    ensure_schema()
    ensure_schema()  # second run: DROP INDEX IF EXISTS is a no-op
    reset_pool()

    with psycopg.connect(e2e_config["db_url"], autocommit=True) as conn:
        names = _index_names(conn)
    assert "idx_retrieval_candidates_event" not in names
    assert {"retrieval_candidates_pkey", "idx_retrieval_candidate_doc"} <= names


def test_primary_key_covers_every_path_the_dropped_index_served(e2e_config):
    """event_id lookups (+ vector_rank order), the event FK cascade and the
    feedback FK all run on the PK once idx_retrieval_candidates_event is gone."""
    import uuid

    from tools.schema import ensure_schema, reset_pool

    ensure_schema()
    reset_pool()

    with psycopg.connect(e2e_config["db_url"], autocommit=True) as conn:
        assert "idx_retrieval_candidates_event" not in _index_names(conn)

        # The PK's leading column is event_id.
        leading = conn.execute(
            """SELECT a.attname
                 FROM pg_index i
                 JOIN pg_attribute a
                   ON a.attrelid = i.indrelid AND a.attnum = i.indkey[0]
                WHERE i.indrelid = 'local.retrieval_candidates'::regclass
                  AND i.indisprimary"""
        ).fetchone()[0]
        assert leading == "event_id"

        events = [str(uuid.uuid4()) for _ in range(40)]
        for event_id in events:
            conn.execute(
                """INSERT INTO local.retrieval_events
                       (id, expires_at, purpose, query_sha256)
                   VALUES (%s::uuid, now() + interval '1 day', 'context_injection', 'x')""",
                (event_id,),
            )
            with conn.cursor() as cur:
                cur.executemany(
                    """INSERT INTO local.retrieval_candidates
                           (event_id, candidate_key, schema_name, doc_id, vector_rank)
                       VALUES (%s::uuid, %s, 'local', %s, %s)""",
                    [(event_id, f"k{rank}", f"obs::{rank}", 100 - rank) for rank in range(100)],
                )
        conn.execute("ANALYZE local.retrieval_candidates")

        with conn.transaction():
            conn.execute("SET LOCAL enable_seqscan = off")
            plan = "\n".join(
                row[0] for row in conn.execute(
                    "EXPLAIN SELECT candidate_key, vector_rank "
                    "FROM local.retrieval_candidates "
                    "WHERE event_id = %s::uuid ORDER BY vector_rank",
                    (events[7],),
                ).fetchall()
            )
        assert "retrieval_candidates_pkey" in plan, plan

        ranked = conn.execute(
            "SELECT vector_rank FROM local.retrieval_candidates "
            "WHERE event_id = %s::uuid ORDER BY vector_rank LIMIT 3",
            (events[7],),
        ).fetchall()
        assert [row[0] for row in ranked] == [1, 2, 3]

        # Candidate feedback FK and the event cascade still work.
        conn.execute(
            """INSERT INTO local.retrieval_candidate_feedback
                   (event_id, candidate_key, verdict)
               VALUES (%s::uuid, 'k5', 'relevant')""",
            (events[0],),
        )
        conn.execute("DELETE FROM local.retrieval_events WHERE id = %s::uuid", (events[0],))
        remaining = conn.execute(
            """SELECT
                 (SELECT count(*) FROM local.retrieval_candidates WHERE event_id = %s::uuid),
                 (SELECT count(*) FROM local.retrieval_candidate_feedback WHERE event_id = %s::uuid)""",
            (events[0], events[0]),
        ).fetchone()
        assert remaining == (0, 0)
