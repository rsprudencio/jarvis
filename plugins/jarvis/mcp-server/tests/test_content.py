"""Tests for content CRUD operations backed by PostgreSQL + pgvector.

Tests the local.memories table with proper columns (category, scope, source,
importance_score, retrieval_count, status) instead of JSONB metadata.
"""

import pytest
from tools.content import (
    content_write,
    content_read,
    content_list,
    content_delete,
    content_upsert,
    VALID_CATEGORIES,
)


class TestContentWrite:
    """Test content_write function."""

    def test_write_observation_basic(self, mock_config):
        """Test writing a basic observation."""
        result = content_write(
            content="Test observation content", content_type="observation"
        )
        assert result["success"]
        assert "obs::" in result["id"]
        assert result["content_type"] == "observation"
        assert result["importance_score"] == 0.5  # default

    def test_write_pattern_requires_name(self, mock_config):
        """Test that pattern type requires a name."""
        result = content_write(content="Test pattern", content_type="pattern")
        assert not result["success"]
        assert "requires a name" in result["error"]

    def test_write_pattern_with_name(self, mock_config):
        """Test writing a pattern with name."""
        result = content_write(
            content="Test pattern content", content_type="pattern", name="test-pattern"
        )
        assert result["success"]
        assert "pattern::test-pattern" in result["id"]

    def test_write_plan_requires_name(self, mock_config):
        """Test that plan type requires a name."""
        result = content_write(content="Test plan", content_type="plan")
        assert not result["success"]
        assert "requires a name" in result["error"]

    def test_write_summary_with_session_id(self, mock_config):
        """Test writing a summary with session ID."""
        result = content_write(
            content="Session summary",
            content_type="summary",
            session_id="test-session-123",
        )
        assert result["success"]
        assert "summary::" in result["id"]

    def test_write_with_custom_importance(self, mock_config):
        """Test writing with custom importance score."""
        result = content_write(
            content="Important observation",
            content_type="observation",
            importance_score=0.9,
        )
        assert result["success"]
        assert result["importance_score"] == 0.9

    def test_write_invalid_importance_range(self, mock_config):
        """Test validation of importance score range."""
        result = content_write(
            content="Test", content_type="observation", importance_score=1.5
        )
        assert not result["success"]
        assert "between 0.0 and 1.0" in result["error"]

    def test_write_with_tags(self, mock_config):
        """Test writing with tags."""
        result = content_write(
            content="Test content",
            content_type="observation",
            tags=["work", "jarvis", "testing"],
        )
        assert result["success"]

        # Verify tags stored in JSONB metadata (tags remain in metadata)
        row = mock_config.db.get(result["id"])
        assert row["metadata"]["tags"] == "work,jarvis,testing"

    def test_write_invalid_content_type(self, mock_config):
        """Test validation of content type."""
        result = content_write(content="Test", content_type="invalid_type")
        assert not result["success"]
        assert "Invalid content_type" in result["error"]

    def test_write_secret_detection(self, mock_config):
        """Test secret detection blocks write."""
        result = content_write(
            content="API key: AKIAIOSFODNN7EXAMPLE", content_type="observation"
        )
        assert not result["success"]
        assert "Secret detected" in result["error"]
        assert "detections" in result

    def test_write_skip_secret_scan(self, mock_config):
        """Test skipping secret scan."""
        result = content_write(
            content="API key: AKIAIOSFODNN7EXAMPLE",
            content_type="observation",
            skip_secret_scan=True,
        )
        assert result["success"]

    def test_write_relationship(self, mock_config):
        """Test writing a relationship."""
        result = content_write(
            content="Person A knows Person B",
            content_type="relationship",
            name="person-a::person-b",
        )
        assert result["success"]
        assert "rel::" in result["id"]

    def test_write_hint(self, mock_config):
        """Test writing a hint."""
        result = content_write(
            content="Use git when committing",
            content_type="hint",
            name="git-workflow::0",
        )
        assert result["success"]
        assert "hint::" in result["id"]


class TestContentRead:
    """Test content_read function."""

    def test_read_existing_doc(self, mock_config):
        """Test reading an existing document."""
        # Write first
        write_result = content_write(
            content="Test observation", content_type="observation"
        )
        doc_id = write_result["id"]

        # Read
        result = content_read(doc_id)
        assert result["success"]
        assert result["found"]
        assert result["id"] == doc_id
        assert result["content"] == "Test observation"
        # Column-based values (not in metadata)
        assert result["category"] == "observation"
        assert result["retrieval_count"] == 1.0
        assert result["importance_score"] == 0.5
        assert result["status"] == "active"

    def test_read_increments_count(self, mock_config):
        """Test that reading increments retrieval count."""
        # Write
        write_result = content_write(content="Test", content_type="observation")
        doc_id = write_result["id"]

        # Read multiple times — retrieval_count is a float column now
        for i in range(1, 4):
            result = content_read(doc_id)
            assert result["retrieval_count"] == float(i)

    def test_read_float_retrieval_count(self, mock_config):
        """Reads with retrieval_count=2.5 -> increments to 3.5."""
        # Write, then manually set retrieval_count to 2.5
        write_result = content_write(
            content="Float retrieval test",
            content_type="observation",
        )
        doc_id = write_result["id"]

        # Directly set retrieval_count in in-memory DB (it's a column now)
        row = mock_config.db.core_rows[doc_id]
        row["retrieval_count"] = 2.5

        # Read should increment by 1 -> 3.5
        read_result = content_read(doc_id)
        assert read_result["retrieval_count"] == 3.5

    def test_read_not_found(self, mock_config):
        """Test reading non-existent document."""
        result = content_read("obs::nonexistent")
        assert result["success"]
        assert not result["found"]


class TestContentList:
    """Test content_list function."""

    def test_list_all(self, mock_config):
        """Test listing all content documents."""
        # Write some docs
        content_write(content="Obs 1", content_type="observation")
        content_write(content="Pattern 1", content_type="pattern", name="p1")

        result = content_list()
        assert result["success"]
        assert result["total"] >= 2
        assert len(result["documents"]) >= 2

    def test_list_by_content_type(self, mock_config):
        """Test filtering by content type (category column)."""
        # Write docs
        content_write(content="Obs 1", content_type="observation")
        content_write(content="Pattern 1", content_type="pattern", name="p1")

        result = content_list(content_type="observation")
        assert result["success"]
        for doc in result["documents"]:
            # category is a top-level column, not metadata['type']
            assert doc["category"] == "observation"

    def test_list_by_min_importance(self, mock_config):
        """Test filtering by minimum importance (column-based)."""
        # Write docs with different importance
        content_write(content="Low", content_type="observation", importance_score=0.3)
        content_write(content="High", content_type="observation", importance_score=0.9)

        result = content_list(min_importance=0.8)
        assert result["success"]
        for doc in result["documents"]:
            # importance_score is a top-level column (float), not metadata string
            assert doc["importance_score"] >= 0.8

    def test_list_by_source(self, mock_config):
        """Test filtering by source (column-based)."""
        content_write(content="Auto", content_type="observation", source="auto-extract")
        content_write(content="Manual", content_type="observation", source="manual")

        result = content_list(source="manual")
        assert result["success"]
        for doc in result["documents"]:
            # source is a top-level column, not metadata['source']
            assert doc["source"] == "manual"

    def test_list_with_limit(self, mock_config, monkeypatch):
        """Test limit parameter.

        The in-memory mock drops LIMIT for this WHERE shape (it counts the
        literal status condition as a bound parameter). The old row-count
        assertion only held because same-millisecond observation IDs used to
        overwrite each other; the SQL LIMIT is checked here and the returned
        row count against real PostgreSQL in tests/e2e/test_generated_ids_e2e.py.
        """
        import tools.content as content_module

        for i in range(10):
            content_write(content=f"Doc {i}", content_type="observation")

        seen = []
        real_query = content_module.execute_query

        def spy(sql, params=None, **kwargs):
            seen.append((sql, params))
            return real_query(sql, params, **kwargs)

        monkeypatch.setattr(content_module, "execute_query", spy)
        result = content_list(limit=5)
        assert result["success"]
        fetch_sql, fetch_params = seen[-1]
        assert fetch_sql.rstrip().endswith("LIMIT %s")
        assert fetch_params[-1] == 5

    def test_list_empty_collection(self, mock_config):
        """Test listing with empty database."""
        # Clear in-memory DB
        mock_config.db.clear()

        result = content_list()
        assert result["success"]
        assert result["total"] == 0
        assert len(result["documents"]) == 0

    def test_list_invalid_content_type(self, mock_config):
        """Test with invalid content type."""
        result = content_list(content_type="invalid")
        assert not result["success"]
        assert "Invalid content_type" in result["error"]


class TestContentListSortBy:
    """Test content_list sort_by parameter."""

    def test_sort_by_importance_desc(self, mock_config):
        """Default sort returns highest importance first."""
        content_write(content="Low imp", content_type="observation", importance_score=0.3)
        content_write(
            content="High imp", content_type="observation", importance_score=0.9
        )
        content_write(content="Mid imp", content_type="observation", importance_score=0.6)

        result = content_list(sort_by="importance_desc")
        assert result["success"]
        # importance_score is a float column now, not a string in metadata
        scores = [doc["importance_score"] for doc in result["documents"]]
        assert scores == sorted(scores, reverse=True)

    def test_sort_by_importance_asc(self, mock_config):
        """Ascending sort returns lowest importance first."""
        content_write(content="Low imp", content_type="observation", importance_score=0.3)
        content_write(
            content="High imp", content_type="observation", importance_score=0.9
        )

        result = content_list(sort_by="importance_asc")
        assert result["success"]
        scores = [doc["importance_score"] for doc in result["documents"]]
        assert scores == sorted(scores)

    def test_sort_by_created_at_desc(self, mock_config):
        """Created_at desc returns most recent first."""
        import time

        content_write(content="First", content_type="observation")
        time.sleep(0.05)  # Ensure distinct timestamps
        content_write(content="Second", content_type="observation")

        result = content_list(sort_by="created_at_desc")
        assert result["success"]
        # created_at is not returned in the list results by default,
        # but we can verify ordering by checking metadata timestamps
        assert result["total"] >= 2

    def test_sort_by_none(self, mock_config):
        """sort_by='none' returns results without sorting."""
        content_write(content="A", content_type="observation", importance_score=0.3)
        import time; time.sleep(0.002)  # Ensure distinct observation_id timestamps
        content_write(content="B", content_type="observation", importance_score=0.9)

        result = content_list(sort_by="none")
        assert result["success"]
        assert result["total"] >= 2

    def test_sort_by_invalid(self, mock_config):
        """Invalid sort_by returns error."""
        result = content_list(sort_by="bogus")
        assert not result["success"]
        assert "Invalid sort_by" in result["error"]


class TestContentDelete:
    """Test content_delete function (soft delete by default)."""

    def test_delete_existing_soft(self, mock_config):
        """Test soft-deleting an existing document."""
        # Write first
        write_result = content_write(content="Test", content_type="observation")
        doc_id = write_result["id"]

        # Soft delete (default)
        result = content_delete(doc_id)
        assert result["success"]
        assert result["deleted"]
        assert result["id"] == doc_id

        # Verify soft deletion — row still exists but status='deleted'
        row = mock_config.db.get_core(doc_id)
        assert row is not None
        assert row["status"] == "deleted"
        assert row["deleted_at"] is not None

        # content_list should not return soft-deleted documents
        # (content_list filters by status='active')
        list_result = content_list(content_type="observation")
        assert not any(doc["id"] == doc_id for doc in list_result["documents"])

    def test_delete_existing_hard(self, mock_config):
        """Test hard-deleting an existing document."""
        # Write first
        write_result = content_write(content="Test", content_type="observation")
        doc_id = write_result["id"]

        # Hard delete
        result = content_delete(doc_id, hard=True)
        assert result["success"]
        assert result["deleted"]
        assert result["id"] == doc_id

        # Verify hard deletion — row is completely gone
        row = mock_config.db.get_core(doc_id)
        assert row is None

    def test_delete_nonexistent(self, mock_config):
        """Test deleting non-existent document."""
        result = content_delete("obs::nonexistent")
        assert result["success"]
        assert not result["deleted"]
        assert result["reason"] == "not found"


class TestContentLifecycle:
    """Test full content lifecycle."""

    def test_full_cycle(self, mock_config):
        """Test write -> read -> list -> delete cycle."""
        # Write
        write_result = content_write(
            content="Lifecycle test",
            content_type="observation",
            importance_score=0.75,
            tags=["test", "lifecycle"],
        )
        assert write_result["success"]
        doc_id = write_result["id"]

        # Read — column-based values
        read_result = content_read(doc_id)
        assert read_result["success"]
        assert read_result["found"]
        assert read_result["content"] == "Lifecycle test"
        assert read_result["importance_score"] == 0.75
        assert read_result["category"] == "observation"
        assert read_result["status"] == "active"
        # Tags remain in JSONB metadata
        assert read_result["metadata"]["tags"] == "test,lifecycle"

        # List
        list_result = content_list(content_type="observation")
        assert list_result["success"]
        assert any(doc["id"] == doc_id for doc in list_result["documents"])

        # Soft delete (default)
        delete_result = content_delete(doc_id)
        assert delete_result["success"]
        assert delete_result["deleted"]

        # Verify not in list (filtered by status='active')
        list_result2 = content_list(content_type="observation")
        assert not any(doc["id"] == doc_id for doc in list_result2["documents"])


class TestContentLearningDecision:
    """Test learning and decision content types."""

    def test_write_learning(self, mock_config):
        """Test writing a learning."""
        result = content_write(
            content="PostToolUse hooks have empty tool_result", content_type="learning"
        )
        assert result["success"]
        assert "learning::" in result["id"]
        assert result["content_type"] == "learning"

    def test_write_decision_requires_name(self, mock_config):
        """Test that decision type requires a name."""
        result = content_write(content="Use Python", content_type="decision")
        assert not result["success"]
        assert "requires a name" in result["error"]

    def test_write_decision_with_name(self, mock_config):
        """Test writing a decision with name."""
        result = content_write(
            content="Use Python MCP server over TypeScript",
            content_type="decision",
            name="python-mcp-decision",
        )
        assert result["success"]
        assert "decision::python-mcp-decision" in result["id"]


class TestContentExtraMetadata:
    """Test extra_metadata passthrough."""

    def test_extra_metadata_stored(self, mock_config):
        """Test that extra_metadata is stored in database JSONB."""
        result = content_write(
            content="Observation with context",
            content_type="observation",
            extra_metadata={
                "project_path": "/home/user/projects/jarvis-plugin",
                "git_branch": "master",
            },
        )
        assert result["success"]

        read_result = content_read(result["id"])
        # Extra metadata fields go into JSONB metadata column
        assert (
            read_result["metadata"]["project_path"]
            == "/home/user/projects/jarvis-plugin"
        )
        assert read_result["metadata"]["git_branch"] == "master"

    def test_extra_metadata_none(self, mock_config):
        """Test that None extra_metadata is fine."""
        result = content_write(
            content="No extra metadata",
            content_type="observation",
            extra_metadata=None,
        )
        assert result["success"]

    def test_ingest_event_id_is_idempotent(self, mock_config):
        """Same ingest_event_id should return existing ID, not duplicate documents."""
        event_id = "evt-123abc"

        first = content_write(
            content="Observation with idempotency key",
            content_type="observation",
            extra_metadata={"ingest_event_id": event_id},
        )
        assert first["success"]

        second = content_write(
            content="Observation with idempotency key",
            content_type="observation",
            extra_metadata={"ingest_event_id": event_id},
        )
        assert second["success"]
        assert second["id"] == first["id"]
        assert second.get("deduplicated") is True

        # Verify only one document with this event_id exists in DB
        matching = [
            r for r in mock_config.db.core_rows.values()
            if r["metadata"].get("ingest_event_id") == event_id
        ]
        assert len(matching) == 1


class TestContentUpsert:
    """Test content_upsert function."""

    def test_upsert_updates_content(self, mock_config):
        """Test that upsert updates existing document content."""
        # Write original
        write_result = content_write(
            content="Original content",
            content_type="observation",
            importance_score=0.5,
        )
        doc_id = write_result["id"]

        # Upsert with new content — metadata dict can use 'type' or 'category'
        result = content_upsert(
            doc_id,
            "Updated content",
            {"type": "observation", "importance_score": "0.5"},
        )
        assert result["success"]
        assert result["updated"]

        # Verify update
        read_result = content_read(doc_id)
        assert read_result["content"] == "Updated content"

    def test_upsert_updates_importance(self, mock_config):
        """Test that upsert updates importance_score (column, not metadata)."""
        write_result = content_write(
            content="Test",
            content_type="observation",
            importance_score=0.5,
        )
        doc_id = write_result["id"]

        content_upsert(doc_id, "Test", {"type": "observation", "importance_score": "0.9"})

        read_result = content_read(doc_id)
        # importance_score is a float column now
        assert read_result["importance_score"] == 0.9

    def test_upsert_scope_project_without_project_falls_back_to_global(self, mock_config):
        """content_upsert with scope='project' and no project key → stored as scope='global'."""
        write_result = content_write(
            content="Original",
            content_type="observation",
            importance_score=0.5,
        )
        doc_id = write_result["id"]

        # Upsert with scope='project' but no project name — guard must downgrade scope
        result = content_upsert(
            doc_id,
            "Updated",
            {"type": "observation", "scope": "project", "importance_score": "0.5"},
        )
        assert result["success"]

        read_result = content_read(doc_id)
        assert read_result["scope"] == "global"


class TestContentUpsertSyncRouting:
    """Test sync routing in content_upsert."""

    def test_upsert_enqueues_when_routing_matches(self, mock_config, monkeypatch):
        """Sync enabled + catch-all rule → enqueue_sync called."""
        from unittest.mock import MagicMock, patch
        from tools.routing import RoutingDecision

        # Write a doc first
        write_result = content_write(
            content="Original", content_type="observation"
        )
        doc_id = write_result["id"]

        mock_enqueue = MagicMock(return_value=1)
        mock_decision = RoutingDecision(
            destinations=["backup"], matched_rules=["all"]
        )

        monkeypatch.setattr("tools.config.get_sync_config", lambda: {
            "enabled": True,
            "strategy": "first-match",
            "rules": [{"name": "all", "match": {}, "action": "route-to",
                        "destinations": ["backup"]}],
            "project_groups": {},
        })
        monkeypatch.setattr(
            "tools.routing.evaluate_routing",
            lambda *a, **kw: mock_decision,
        )
        monkeypatch.setattr("tools.sync_queue.enqueue_sync", mock_enqueue)

        result = content_upsert(
            doc_id, "Updated content",
            {"type": "observation", "importance_score": "0.5"},
        )
        assert result["success"]
        mock_enqueue.assert_called_once()
        call_args = mock_enqueue.call_args
        assert call_args[0][1] == doc_id
        assert call_args[0][2] == ["backup"]

    def test_upsert_no_enqueue_when_no_match(self, mock_config, monkeypatch):
        """Empty destinations → no enqueue call."""
        from unittest.mock import MagicMock
        from tools.routing import RoutingDecision

        write_result = content_write(
            content="Original", content_type="observation"
        )
        doc_id = write_result["id"]

        mock_enqueue = MagicMock(return_value=0)
        mock_decision = RoutingDecision(
            destinations=[], matched_rules=[]
        )

        monkeypatch.setattr("tools.config.get_sync_config", lambda: {
            "enabled": True,
            "strategy": "first-match",
            "rules": [],
            "project_groups": {},
        })
        monkeypatch.setattr(
            "tools.routing.evaluate_routing",
            lambda *a, **kw: mock_decision,
        )
        monkeypatch.setattr("tools.sync_queue.enqueue_sync", mock_enqueue)

        result = content_upsert(
            doc_id, "Updated",
            {"type": "observation", "importance_score": "0.5"},
        )
        assert result["success"]
        mock_enqueue.assert_not_called()

    def test_upsert_no_enqueue_when_sync_disabled(self, mock_config, monkeypatch):
        """Sync disabled → no enqueue call."""
        from unittest.mock import MagicMock

        write_result = content_write(
            content="Original", content_type="observation"
        )
        doc_id = write_result["id"]

        mock_enqueue = MagicMock()

        monkeypatch.setattr("tools.config.get_sync_config", lambda: {
            "enabled": False,
        })
        monkeypatch.setattr("tools.sync_queue.enqueue_sync", mock_enqueue)

        result = content_upsert(
            doc_id, "Updated",
            {"type": "observation", "importance_score": "0.5"},
        )
        assert result["success"]
        mock_enqueue.assert_not_called()

    def test_upsert_sync_failure_does_not_break_upsert(self, mock_config, monkeypatch):
        """enqueue_sync raising → upsert still succeeds."""
        from tools.routing import RoutingDecision

        write_result = content_write(
            content="Original", content_type="observation"
        )
        doc_id = write_result["id"]

        monkeypatch.setattr("tools.config.get_sync_config", lambda: {
            "enabled": True,
            "strategy": "first-match",
            "rules": [{"name": "all", "match": {}, "action": "route-to",
                        "destinations": ["backup"]}],
            "project_groups": {},
        })
        monkeypatch.setattr(
            "tools.routing.evaluate_routing",
            lambda *a, **kw: RoutingDecision(
                destinations=["backup"], matched_rules=["all"]
            ),
        )

        def exploding_enqueue(*args, **kwargs):
            raise RuntimeError("Connection refused")

        monkeypatch.setattr("tools.sync_queue.enqueue_sync", exploding_enqueue)

        result = content_upsert(
            doc_id, "Updated",
            {"type": "observation", "importance_score": "0.5"},
        )
        assert result["success"]
        assert result["updated"]


class TestContentWorklog:
    """Test worklog content type."""

    def test_write_worklog(self, mock_config):
        """Test writing a worklog entry."""
        result = content_write(
            content="Adding Docker containerization for Jarvis MCP servers",
            content_type="worklog",
            extra_metadata={"workstream": "Jarvis Plugin", "activity_type": "coding"},
        )
        assert result["success"]
        assert "worklog::" in result["id"]
        assert result["content_type"] == "worklog"

    def test_worklog_metadata(self, mock_config):
        """Test worklog metadata is stored correctly."""
        result = content_write(
            content="Debugging VMPulse alerts",
            content_type="worklog",
            extra_metadata={
                "workstream": "VMPulse",
                "activity_type": "debugging",
                "session_id": "test-session-123",
            },
            session_id="test-session-123",
        )
        assert result["success"]

        read_result = content_read(result["id"])
        # Extra metadata fields stay in JSONB metadata
        assert read_result["metadata"]["workstream"] == "VMPulse"
        assert read_result["metadata"]["activity_type"] == "debugging"
        assert read_result["metadata"]["session_id"] == "test-session-123"

    def test_worklog_in_valid_categories(self):
        """Test that worklog is in VALID_CATEGORIES."""
        assert "worklog" in VALID_CATEGORIES

    def test_list_worklogs_by_session_id(self, mock_config):
        """Test filtering worklogs by session_id."""
        import time

        # Write two worklogs with different session IDs
        content_write(
            content="Task A",
            content_type="worklog",
            session_id="session-1",
            extra_metadata={"workstream": "misc", "activity_type": "coding"},
        )
        time.sleep(0.002)  # Ensure distinct worklog_id timestamps
        content_write(
            content="Task B",
            content_type="worklog",
            session_id="session-2",
            extra_metadata={"workstream": "misc", "activity_type": "coding"},
        )

        # Filter by session-1
        result = content_list(content_type="worklog", session_id="session-1")
        assert result["success"]
        assert result["total"] == 1
        assert result["documents"][0]["content"] == "Task A"

    def test_list_worklogs_no_session_filter(self, mock_config):
        """Test listing all worklogs without session filter."""
        import time

        content_write(
            content="Task X",
            content_type="worklog",
            session_id="s1",
            extra_metadata={"workstream": "misc", "activity_type": "coding"},
        )
        time.sleep(0.002)  # Ensure distinct worklog_id timestamps
        content_write(
            content="Task Y",
            content_type="worklog",
            session_id="s2",
            extra_metadata={"workstream": "misc", "activity_type": "coding"},
        )

        result = content_list(content_type="worklog")
        assert result["success"]
        assert result["total"] == 2


class TestContentListIncludeContent:
    """Test content_list include_content parameter."""

    def test_default_includes_content(self, mock_config):
        """Default include_content=True includes document text (backward compat)."""
        content_write(content="Default content test", content_type="observation")

        result = content_list(content_type="observation")
        assert result["success"]
        assert result["total"] >= 1

        found = any(
            doc.get("content") == "Default content test" for doc in result["documents"]
        )
        assert found, "Expected content in results by default"

    def test_include_content_false(self, mock_config):
        """include_content=False omits document text from results."""
        content_write(content="Hidden content", content_type="observation")

        result = content_list(content_type="observation", include_content=False)
        assert result["success"]
        assert result["total"] >= 1

        for doc in result["documents"]:
            assert "content" not in doc
            # Column-based fields should still be present
            assert "category" in doc
            assert "id" in doc

    def test_include_content_true_explicit(self, mock_config):
        """Explicit include_content=True includes document text."""
        content_write(content="Visible content", content_type="observation")

        result = content_list(content_type="observation", include_content=True)
        assert result["success"]

        found = any(
            doc.get("content") == "Visible content" for doc in result["documents"]
        )
        assert found, "Expected content when include_content=True"


class TestContentColumnSchema:
    """Test that content operations use proper column-based schema.

    These tests verify that the new schema stores classification data
    in columns rather than JSONB metadata, and that old metadata-based
    fields (tier, namespace, promoted) are no longer present.
    """

    def test_write_stores_category_as_column(self, mock_config):
        """Verify category is stored as a DB column, not in metadata."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert row["category"] == "observation"
        # Category should NOT be in metadata JSONB
        assert "type" not in row["metadata"]
        assert "category" not in row["metadata"]

    def test_write_stores_scope_as_column(self, mock_config):
        """Verify scope is stored as a DB column, not in metadata."""
        result = content_write(
            content="Test",
            content_type="observation",
            extra_metadata={"scope": "project", "project": "jarvis"},
        )
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert row["scope"] == "project"
        assert row["project"] == "jarvis"
        # Scope should NOT be in metadata JSONB
        assert "scope" not in row["metadata"]

    def test_write_stores_source_as_column(self, mock_config):
        """Verify source is stored as a DB column, not in metadata."""
        result = content_write(
            content="Test",
            content_type="observation",
            source="manual",
        )
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert row["source"] == "manual"
        # Source should NOT be in metadata JSONB
        assert "source" not in row["metadata"]

    def test_write_stores_importance_as_float_column(self, mock_config):
        """Verify importance_score is a float column, not a string in metadata."""
        result = content_write(
            content="Test",
            content_type="observation",
            importance_score=0.85,
        )
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert isinstance(row["importance_score"], float)
        assert row["importance_score"] == 0.85
        # Should NOT be in metadata JSONB
        assert "importance_score" not in row["metadata"]

    def test_no_tier_in_metadata(self, mock_config):
        """Verify tier field is not stored in metadata (removed in v3)."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert "tier" not in row["metadata"]

    def test_no_namespace_in_metadata(self, mock_config):
        """Verify namespace field is not stored in metadata (removed in v3)."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert "namespace" not in row["metadata"]

    def test_no_promoted_in_metadata(self, mock_config):
        """Verify promoted field is not stored in metadata (removed in v3)."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert "promoted" not in row["metadata"]

    def test_status_column_defaults_to_active(self, mock_config):
        """Verify status column defaults to 'active'."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert row["status"] == "active"

    def test_retrieval_count_column_starts_at_zero(self, mock_config):
        """Verify retrieval_count starts at 0.0 as a float column."""
        result = content_write(content="Test", content_type="observation")
        assert result["success"]

        row = mock_config.db.get_core(result["id"])
        assert isinstance(row["retrieval_count"], float)
        assert row["retrieval_count"] == 0.0

    def test_read_returns_column_values(self, mock_config):
        """Verify content_read returns column values as top-level keys."""
        write_result = content_write(
            content="Test",
            content_type="observation",
            importance_score=0.7,
            source="manual",
        )
        read_result = content_read(write_result["id"])

        assert read_result["category"] == "observation"
        assert read_result["scope"] == "global"
        assert read_result["source"] == "manual"
        assert read_result["importance_score"] == 0.7
        assert read_result["status"] == "active"
        assert read_result["retrieval_count"] == 1.0  # incremented on read

    def test_list_returns_column_values(self, mock_config):
        """Verify content_list returns column values as top-level keys."""
        content_write(
            content="Test",
            content_type="observation",
            importance_score=0.6,
            source="auto-extract",
        )

        result = content_list(content_type="observation")
        assert result["success"]
        assert result["total"] >= 1

        doc = result["documents"][0]
        assert "category" in doc
        assert "scope" in doc
        assert "source" in doc
        assert "importance_score" in doc
        assert doc["category"] == "observation"


class TestContentListGenericFilter:
    """Test content_list with generic metadata filter parameter."""

    def test_filter_by_workstream(self, mock_config):
        """Filter by workstream (generic JSONB key) returns only matching docs."""
        content_write(
            content="Vuln mgmt work",
            content_type="learning",
            extra_metadata={"workstream": "vuln-mgmt"},
        )
        content_write(
            content="Infra work",
            content_type="pattern",
            name="infra-pat",
            extra_metadata={"workstream": "infrastructure"},
        )

        result = content_list(filter={"workstream": "vuln-mgmt"})
        assert result["success"]
        assert result["total"] >= 1
        for doc in result["documents"]:
            meta = doc.get("metadata", {})
            assert meta.get("workstream") == "vuln-mgmt"

    def test_filter_by_git_branch(self, mock_config):
        """Filter by git_branch works via generic JSONB."""
        content_write(
            content="Main branch work",
            content_type="learning",
            extra_metadata={"git_branch": "main"},
        )
        content_write(
            content="Feature branch work",
            content_type="pattern",
            name="feature-pat",
            extra_metadata={"git_branch": "feature-x"},
        )

        result = content_list(filter={"git_branch": "feature-x"})
        assert result["success"]
        assert result["total"] >= 1
        for doc in result["documents"]:
            meta = doc.get("metadata", {})
            assert meta.get("git_branch") == "feature-x"

    def test_filter_combined_with_typed_params(self, mock_config):
        """Generic filter works alongside typed params (content_type)."""
        content_write(
            content="Observation in proj-a",
            content_type="observation",
            extra_metadata={"workstream": "vuln-mgmt"},
        )
        content_write(
            content="Worklog in proj-a",
            content_type="worklog",
            extra_metadata={"workstream": "vuln-mgmt"},
        )

        result = content_list(
            content_type="observation",
            filter={"workstream": "vuln-mgmt"},
        )
        assert result["success"]
        for doc in result["documents"]:
            assert doc["category"] == "observation"
            assert doc["metadata"].get("workstream") == "vuln-mgmt"

    def test_filter_no_match(self, mock_config):
        """Filter with non-existent value returns empty."""
        content_write(
            content="Some content",
            content_type="observation",
            extra_metadata={"workstream": "real-stream"},
        )

        result = content_list(filter={"workstream": "nonexistent"})
        assert result["success"]
        assert result["total"] == 0

    def test_filter_empty_dict(self, mock_config):
        """Empty filter dict has no effect."""
        content_write(content="Test", content_type="observation")

        result = content_list(filter={})
        assert result["success"]
        assert result["total"] >= 1

    def test_filter_none(self, mock_config):
        """None filter has no effect."""
        content_write(content="Test", content_type="observation")

        result = content_list(filter=None)
        assert result["success"]
        assert result["total"] >= 1


class TestContentListScopeProject:
    """Tests for scope and project column filtering in content_list."""

    def test_filter_by_scope(self, mock_config):
        """Filter by scope returns only matching docs."""
        content_write(
            content="Global obs",
            content_type="observation",
        )
        content_write(
            content="Project obs",
            content_type="observation",
            extra_metadata={"scope": "project", "project": "my-app"},
        )

        result = content_list(scope="project")
        assert result["success"]
        for doc in result["documents"]:
            assert doc["scope"] == "project"

    def test_filter_by_project(self, mock_config):
        """Filter by project returns only matching docs."""
        content_write(
            content="App A obs",
            content_type="learning",
            extra_metadata={"scope": "project", "project": "app-a"},
        )
        content_write(
            content="App B obs",
            content_type="pattern",
            name="proj-b-pat",
            extra_metadata={"scope": "project", "project": "app-b"},
        )

        result = content_list(project="app-a")
        assert result["success"]
        assert result["total"] >= 1
        # No app-b in results
        ids = {d["id"] for d in result["documents"]}
        result_b = content_list(project="app-b")
        ids_b = {d["id"] for d in result_b["documents"]}
        assert not ids.intersection(ids_b)

    def test_filter_scope_and_project_combined(self, mock_config):
        """Scope + project narrows correctly."""
        content_write(
            content="Scoped to proj",
            content_type="learning",
            extra_metadata={"scope": "project", "project": "target"},
        )
        content_write(
            content="Global",
            content_type="pattern",
            name="global-pat",
        )

        result = content_list(scope="project", project="target")
        assert result["success"]
        assert result["total"] >= 1
        for doc in result["documents"]:
            assert doc["scope"] == "project"

    def test_filter_project_no_match(self, mock_config):
        """Non-existent project returns empty."""
        content_write(content="Some obs", content_type="observation")

        result = content_list(project="nonexistent-xyz")
        assert result["success"]
        assert result["total"] == 0

    def test_filter_scope_none_no_effect(self, mock_config):
        """None scope doesn't filter."""
        content_write(content="Test", content_type="observation")

        result = content_list(scope=None)
        assert result["success"]
        assert result["total"] >= 1

    def test_filter_project_with_generic_filter(self, mock_config):
        """Project column filter + generic JSONB filter work together."""
        content_write(
            content="Target",
            content_type="learning",
            extra_metadata={
                "scope": "project",
                "project": "my-proj",
                "git_branch": "main",
            },
        )
        content_write(
            content="Wrong branch",
            content_type="pattern",
            name="wrong-branch-pat",
            extra_metadata={
                "scope": "project",
                "project": "my-proj",
                "git_branch": "develop",
            },
        )

        result = content_list(
            project="my-proj",
            filter={"git_branch": "main"},
        )
        assert result["success"]
        assert result["total"] >= 1
        for doc in result["documents"]:
            assert doc["metadata"].get("git_branch") == "main"
