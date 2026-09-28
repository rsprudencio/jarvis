from fastapi.testclient import TestClient

import app as app_module


def test_health_reports_ready_sources():
    original = app_module._sources
    app_module._sources = {"local": {}, "obsidian": {}}
    try:
        response = TestClient(app_module.app).get("/health")
    finally:
        app_module._sources = original

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "server": "memory-explorer",
        "sources": ["local", "obsidian"],
        # No lifespan ran, so the background DB probe never filled the cache.
        "postgres": {"status": "unknown", "error": None, "checked_at": None, "free_bytes": None},
    }
