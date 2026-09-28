"""Unit tests for tools.hook_endpoints."""

from unittest.mock import MagicMock

import tools.hook_endpoints as hook_endpoints


def test_get_prompt_context_disabled(monkeypatch):
    """Disabled config returns empty context and skips search."""
    monkeypatch.setattr(
        hook_endpoints,
        "get_context_enrichment_config",
        lambda: {"enabled": False, "budget": 9000, "debug": True},
    )
    monkeypatch.setattr(
        hook_endpoints,
        "get_todoist_prompt_alerts_config",
        lambda: {"enabled": True, "max_per_category": 2},
    )
    semantic = MagicMock()
    monkeypatch.setattr(hook_endpoints, "semantic_context", semantic)

    result = hook_endpoints.get_prompt_context("plan next sprint")

    assert result["success"] is True
    assert result["enabled"] is False
    assert result["matches"] == []
    assert result["budget_used"]["total"] == 9000
    assert result["todoist_prompt_alerts"]["enabled"] is True
    semantic.assert_not_called()


def test_get_prompt_context_with_matches(monkeypatch):
    """Prompt context delegates to semantic_context and returns matches."""
    monkeypatch.setattr(
        hook_endpoints,
        "get_context_enrichment_config",
        lambda: {"enabled": True, "threshold": 0.6, "budget": 1200, "debug": False},
    )
    monkeypatch.setattr(
        hook_endpoints,
        "get_todoist_prompt_alerts_config",
        lambda: {"enabled": False, "max_per_category": 3},
    )

    semantic = MagicMock(
        return_value={
            "matches": [{"id": "notes/a.md", "relevance": 0.82}],
            "query_ms": 7.2,
            "total_searched": 4,
            "semantic_duplicates_suppressed": 2,
            "budget_used": {"local": 200, "vault": 300, "remote": 0, "total": 1200},
        }
    )
    monkeypatch.setattr(hook_endpoints, "semantic_context", semantic)

    result = hook_endpoints.get_prompt_context("what are my goals")

    semantic.assert_called_once_with(
        query="what are my goals",
        threshold=0.6,
        budget=1200,
        skip_retrieval_increment=False,
        schemas=None,  # default "all" → _parse_schemas returns None
        max_results=20,
    )
    assert result["success"] is True
    assert len(result["matches"]) == 1
    assert result["query_ms"] == 7.2
    assert result["total_searched"] == 4
    assert result["semantic_duplicates_suppressed"] == 2


def test_ingest_auto_extract_observation_dedup_threshold(monkeypatch):
    """Observation dedup threshold controls whether write occurs."""
    monkeypatch.setattr(
        hook_endpoints,
        "query_vault",
        lambda **kwargs: {
            "success": True,
            "results": [{"relevance": 0.97, "preview": "existing"}],
        },
    )
    monkeypatch.setattr(
        hook_endpoints,
        "content_list",
        lambda **kwargs: {"success": True, "documents": []},
    )

    write = MagicMock(return_value={"success": True, "id": "obs::new"})
    monkeypatch.setattr(hook_endpoints, "content_write", write)

    payload = {
        "observations": [{"content": "Same idea", "importance_score": 0.6, "tags": []}],
        "worklog": None,
        "context": {"session_id": "s1", "transcript_line": 10},
        "dedup": {"observation_threshold": 0.95, "worklog_threshold": 0.7},
    }
    dedup_result = hook_endpoints.ingest_auto_extract(payload)
    assert dedup_result["observations"][0]["status"] == "duplicate"
    write.assert_not_called()

    payload["dedup"]["observation_threshold"] = 0.99
    store_result = hook_endpoints.ingest_auto_extract(payload)
    assert store_result["observations"][0]["status"] == "stored"
    write.assert_called_once()


def test_ingest_auto_extract_worklog_dedup_by_session(monkeypatch):
    """Worklog dedup uses session-filtered documents and Jaccard threshold."""
    monkeypatch.setattr(
        hook_endpoints,
        "query_vault",
        lambda **kwargs: {"success": True, "results": []},
    )
    content_list = MagicMock(
        return_value={
            "success": True,
            "documents": [{"content": "Adding docker support to jarvis plugin"}],
        }
    )
    monkeypatch.setattr(hook_endpoints, "content_list", content_list)

    write = MagicMock(return_value={"success": True, "id": "worklog::123"})
    monkeypatch.setattr(hook_endpoints, "content_write", write)

    payload = {
        "observations": [],
        "worklog": {
            "task_summary": "Adding docker support to jarvis plugin",
            "workstream": "Jarvis Plugin",
            "activity_type": "coding",
            "tags": [],
        },
        "context": {"session_id": "session-42", "transcript_line": 30},
        "dedup": {"observation_threshold": 0.95, "worklog_threshold": 0.6},
    }
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["worklog"]["status"] == "duplicate"
    content_list.assert_called_once_with(
        content_type="worklog",
        limit=20,
        sort_by="created_at_desc",
    )
    write.assert_not_called()


def test_ingest_event_id_passthrough(monkeypatch):
    """ingest_event_id is forwarded into content_write extra_metadata."""
    monkeypatch.setattr(
        hook_endpoints,
        "query_vault",
        lambda **kwargs: {"success": True, "results": []},
    )
    monkeypatch.setattr(
        hook_endpoints,
        "content_list",
        lambda **kwargs: {"success": True, "documents": []},
    )

    calls = []

    def _fake_content_write(**kwargs):
        calls.append(kwargs)
        content_type = kwargs["content_type"]
        doc_id = "obs::1" if content_type == "observation" else "worklog::1"
        return {"success": True, "id": doc_id}

    monkeypatch.setattr(hook_endpoints, "content_write", _fake_content_write)

    payload = {
        "observations": [
            {
                "content": "User prefers concise answers",
                "importance_score": 0.6,
                "tags": ["style"],
                "scope": "global",
                "ingest_event_id": "obs:event:123",
            }
        ],
        "worklog": {
            "task_summary": "Refactor hook transport",
            "workstream": "Jarvis Plugin",
            "activity_type": "coding",
            "tags": ["hooks"],
            "ingest_event_id": "worklog:event:456",
        },
        "context": {
            "project_path": "/tmp/project",
            "git_branch": "main",
            "relevant_files": ["a.py"],
            "file_mtimes": {"a.py": 123.4},
            "session_id": "s-1",
            "transcript_line": 99,
        },
        "dedup": {"observation_threshold": 0.95, "worklog_threshold": 0.7},
    }
    result = hook_endpoints.ingest_auto_extract(payload)
    assert result["success"] is True
    assert len(calls) == 2

    obs_call = next(call for call in calls if call["content_type"] == "observation")
    wl_call = next(call for call in calls if call["content_type"] == "worklog")
    assert obs_call["extra_metadata"]["ingest_event_id"] == "obs:event:123"
    assert wl_call["extra_metadata"]["ingest_event_id"] == "worklog:event:456"


def _make_obs_payload(scope, project_path=None, raw_project=None, relevant_files=None):
    """Helper: build a minimal ingest_auto_extract payload for one observation."""
    obs = {
        "content": "Some project-specific insight",
        "importance_score": 0.7,
        "scope": scope,
    }
    if raw_project is not None:
        obs["project"] = raw_project
    context = {}
    if project_path is not None:
        context["project_path"] = project_path
    if relevant_files is not None:
        context["relevant_files"] = relevant_files
    return {
        "observations": [obs],
        "worklog": None,
        "context": context,
        "dedup": {"observation_threshold": 0.0, "worklog_threshold": 0.95},
    }


def test_observation_project_scope_uses_project_path(monkeypatch):
    """scope='project' with no raw.project → project set from context project_path basename."""
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kwargs: {"success": True, "results": []}
    )
    calls = []
    monkeypatch.setattr(
        hook_endpoints,
        "content_write",
        lambda **kwargs: calls.append(kwargs) or {"success": True, "id": "obs::1"},
    )

    payload = _make_obs_payload(scope="project", project_path="/home/user/myrepo")
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["success"] is True
    assert len(calls) == 1
    meta = calls[0]["extra_metadata"]
    assert meta.get("scope") == "project"
    assert meta.get("project") == "myrepo"


def test_observation_project_scope_prefers_raw_project(monkeypatch):
    """scope='project' with explicit raw.project → that value overrides path basename."""
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kwargs: {"success": True, "results": []}
    )
    calls = []
    monkeypatch.setattr(
        hook_endpoints,
        "content_write",
        lambda **kwargs: calls.append(kwargs) or {"success": True, "id": "obs::1"},
    )

    payload = _make_obs_payload(
        scope="project",
        project_path="/home/user/myrepo",
        raw_project="explicit-project-name",
    )
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["success"] is True
    assert len(calls) == 1
    meta = calls[0]["extra_metadata"]
    assert meta.get("project") == "explicit-project-name"


def test_observation_project_scope_no_path_no_project(monkeypatch):
    """scope='project' with no project_path and no raw.project → project not set."""
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kwargs: {"success": True, "results": []}
    )
    calls = []
    monkeypatch.setattr(
        hook_endpoints,
        "content_write",
        lambda **kwargs: calls.append(kwargs) or {"success": True, "id": "obs::1"},
    )

    payload = _make_obs_payload(scope="project")
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["success"] is True
    assert len(calls) == 1
    meta = calls[0]["extra_metadata"]
    assert meta.get("scope") == "project"
    assert "project" not in meta


def test_observation_project_scope_kept_when_files_outside_project(monkeypatch):
    """scope='project' is preserved even when relevant_files are outside project_path.

    Since v3.1.4, scope is trusted from the extraction pipeline — no
    file-path-based downgrade at ingest time.
    """
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kwargs: {"success": True, "results": []}
    )
    calls = []
    monkeypatch.setattr(
        hook_endpoints,
        "content_write",
        lambda **kwargs: calls.append(kwargs) or {"success": True, "id": "obs::1"},
    )

    # Working dir is personio-framework, but files are in jarvis-plugin
    payload = _make_obs_payload(
        scope="project",
        project_path="/home/user/dev/personio-framework",
        relevant_files=[
            "/home/user/0x88/jarvis-plugin/tools/query.py",
            "/home/user/0x88/jarvis-plugin/tools/schema.py",
        ],
    )
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["success"] is True
    assert len(calls) == 1
    meta = calls[0]["extra_metadata"]
    # Scope stays project — trusted from extraction pipeline
    assert meta.get("scope") == "project"
    assert meta.get("project") == "personio-framework"


def test_observation_project_scope_kept_when_files_inside_project(monkeypatch):
    """scope='project' stays when relevant_files are inside project_path."""
    monkeypatch.setattr(
        hook_endpoints, "query_vault", lambda **kwargs: {"success": True, "results": []}
    )
    calls = []
    monkeypatch.setattr(
        hook_endpoints,
        "content_write",
        lambda **kwargs: calls.append(kwargs) or {"success": True, "id": "obs::1"},
    )

    payload = _make_obs_payload(
        scope="project",
        project_path="/home/user/dev/personio-framework",
        relevant_files=[
            "/home/user/dev/personio-framework/src/main.py",
            "/home/user/dev/personio-framework/tests/test.py",
        ],
    )
    result = hook_endpoints.ingest_auto_extract(payload)

    assert result["success"] is True
    assert len(calls) == 1
    meta = calls[0]["extra_metadata"]
    assert meta.get("scope") == "project"
    assert meta.get("project") == "personio-framework"


def test_observation_dedup_gates_on_similarity_not_relevance(monkeypatch):
    """Dedup is a similarity question: an importance-boosted relevance above
    the threshold must NOT trigger dedup when raw similarity is below it."""
    monkeypatch.setattr(
        hook_endpoints,
        "query_vault",
        lambda **kwargs: {
            "success": True,
            "results": [{"similarity": 0.93, "relevance": 0.97, "preview": "x"}],
        },
    )
    assert hook_endpoints._is_duplicate_observation("content", 0.95) is False


def test_observation_dedup_takes_max_similarity_across_window(monkeypatch):
    """query_vault orders by relevance, so the nearest-by-similarity candidate
    can rank below fresher entries — the gate must scan the whole candidate
    window for the max similarity, not just the top-by-relevance hit."""
    captured = {}

    def fake_query_vault(**kwargs):
        captured.update(kwargs)
        return {
            "success": True,
            "results": [
                # Fresh unrelated observation: top by relevance, sim below gate
                {"similarity": 0.90, "relevance": 1.02},
                # Stale true duplicate: dragged down by staleness penalty
                {"similarity": 0.97, "relevance": 0.71},
            ],
        }

    monkeypatch.setattr(hook_endpoints, "query_vault", fake_query_vault)
    assert hook_endpoints._is_duplicate_observation("content", 0.95) is True
    assert captured.get("n_results", 1) > 1, (
        "dedup must fetch a candidate window, not a single top-relevance hit"
    )
    # Similarity is the dedup signal; a host rerank per observation pushed
    # healthy ingests past the 2s hook deadline.
    assert captured.get("rerank") is False
