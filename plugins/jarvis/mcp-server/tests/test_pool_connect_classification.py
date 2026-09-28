"""Why a checkout failed: configure errors and TCP settings (review F3, F2).

_TrackedConnection records a connect as successful before the pool's
configure step (register_vector) runs. A database without the vector
extension then failed every checkout with "connection pool busy" after the
full timeout and never tripped the breaker.
"""

from __future__ import annotations

import psycopg
import psycopg_pool
import pytest

import tools.schema as schema


@pytest.fixture(autouse=True)
def clean_breaker():
    schema.reset_breaker()
    yield
    schema.reset_breaker()


def test_configure_failure_is_recorded_as_a_connect_error(monkeypatch):
    import pgvector.psycopg

    def no_vector_type(conn):
        raise psycopg.ProgrammingError("vector type not found in the database")

    monkeypatch.setattr(pgvector.psycopg, "register_vector", no_vector_type)
    with pytest.raises(psycopg.ProgrammingError):
        schema._configure_connection(object())

    # The pool then times out with nothing handed out: an outage, not contention.
    err = schema._checkout_failed(psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec"))
    assert str(err) == "vector type not found in the database"
    assert schema.db_available() is False


def test_configure_success_leaves_state_alone(monkeypatch):
    import pgvector.psycopg

    seen = []
    monkeypatch.setattr(pgvector.psycopg, "register_vector", seen.append)
    conn = object()
    schema._configure_connection(conn)
    assert seen == [conn]
    err = schema._checkout_failed(psycopg_pool.PoolTimeout("couldn't get a connection after 1.50 sec"))
    assert str(err).startswith("connection pool busy")
    assert schema.db_available() is True


def test_pool_uses_the_recording_configure_step(monkeypatch):
    import tools.config as config

    monkeypatch.setattr(config, "get_postgres_config", lambda: {"url": "postgresql://u@h/db"})
    monkeypatch.setattr(config, "get_embedding_config", lambda: {"dimensions": 384})
    monkeypatch.setattr(config, "get_memory_config", lambda: {})
    created = []

    class FakePool:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def close(self):
            pass

    monkeypatch.setattr(schema, "_JarvisPool", FakePool)
    monkeypatch.setattr(schema, "_pool", None)
    monkeypatch.setattr(schema, "_pool_cache_key", None)
    schema._get_pool()
    kwargs = created[0]
    assert kwargs["configure"] is schema._configure_connection
    # A vanished peer (VM paused, network gone) is detected instead of waited on.
    tcp = kwargs["kwargs"]
    assert tcp["keepalives"] == 1
    assert (tcp["keepalives_idle"], tcp["keepalives_interval"], tcp["keepalives_count"]) == (30, 10, 3)
    if psycopg.pq.version() >= 120000:
        assert tcp["tcp_user_timeout"] == 30_000
    monkeypatch.setattr(schema, "_pool", None)
    monkeypatch.setattr(schema, "_pool_cache_key", None)


def test_tcp_settings_are_accepted_by_libpq():
    """Every key must be a valid conninfo option, or no connection would open."""
    from psycopg.conninfo import make_conninfo

    info = make_conninfo("postgresql://u@127.0.0.1:1/db", **schema._POOL_CONNECT_KWARGS)
    assert "keepalives_idle=30" in info


def test_stalled_probe_opens_the_breaker():
    """A server that accepts and then never answers (frozen, not crash-looping).

    Without the breaker a hook deadline miss reads as "slow request", and the
    hook clients would keep paying their full deadline on every prompt.
    """
    schema.mark_db_probe_stalled(5)
    assert schema.db_available() is False
    status = schema.get_db_status()
    assert status["status"] == "unreachable"
    assert "did not answer the status probe within 5s" in status["error"]
    assert "did not answer" in schema.db_unavailable_reason()
