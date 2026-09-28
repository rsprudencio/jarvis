"""Tests for memory query module."""

import os
import pytest
from unittest.mock import MagicMock, patch
from tools.query import (
    query_vault,
    doc_read,
    collection_stats,
    memory_read,
    memory_stats,  # backward-compatible aliases
    _extract_preview,
    _display_path,
    _increment_retrieval_counts,
    _parse_optional_row_datetime,
    _build_core_filter,
    _build_vault_filter,
    _semantic_deduplicate_context,
    _search_query_windows,
    semantic_context,
)
from tools.ranking import compute_unified_score


def test_missing_optional_datetime_stays_none():
    """Absent retrieval timestamps must not look freshly retrieved."""
    assert _parse_optional_row_datetime(None) is None
    assert _parse_optional_row_datetime("not-a-date") is None


class TestExtractPreview:
    """Tests for content preview extraction."""

    def test_strips_frontmatter(self):
        content = (
            "---\ntype: note\nimportance: high\n---\n# Title\n\nActual content here."
        )
        preview = _extract_preview(content)
        assert "---" not in preview
        assert "type: note" not in preview
        assert "Actual content here." in preview

    def test_strips_heading(self):
        content = "# My Heading\n\nBody text follows."
        preview = _extract_preview(content)
        assert "My Heading" not in preview
        assert "Body text follows." in preview

    def test_truncates_at_word_boundary(self):
        content = "A " * 100  # 200 chars
        preview = _extract_preview(content, max_len=150)
        assert len(preview) <= 154  # max_len + "..."
        assert preview.endswith("...")

    def test_short_content_not_truncated(self):
        content = "Short text."
        preview = _extract_preview(content)
        assert preview == "Short text."
        assert "..." not in preview

    def test_collapses_whitespace(self):
        content = "Line one.\n\n\nLine two.\n\nLine three."
        preview = _extract_preview(content)
        assert "  " not in preview


class TestDisplayPath:
    """Tests for namespace stripping in display paths."""

    def test_vault_prefix_stripped(self):
        assert _display_path("vault::notes/test.md") == "notes/test.md"

    def test_bare_id_unchanged(self):
        assert _display_path("notes/test.md") == "notes/test.md"

    def test_memory_prefix_stripped(self):
        assert _display_path("memory::global::jarvis-trajectory") == "jarvis-trajectory"

    def test_obs_prefix_stripped(self):
        assert _display_path("obs::1738857000000") == "1738857000000"


def test_large_query_searches_every_bounded_window(monkeypatch):
    from tools import query as query_module

    monkeypatch.setattr(query_module, "_QUERY_WINDOW_TOKENS", 4)
    monkeypatch.setattr(query_module, "_QUERY_WINDOW_OVERLAP", 1)
    service = MagicMock()
    service.tokenize.side_effect = lambda text, with_pieces=False: [
        {"id": index, "piece": char} for index, char in enumerate(text)
    ]
    service.encode_batch.return_value = [[1.0], [2.0], [3.0]]

    def search(vector, *_args, **_kwargs):
        value = int(vector[0])
        return [
            {
                "id": f"obs::{value}",
                "_schema": "local",
                "distance": 0.1,
                "document": f"document {value}",
            }
        ]

    with patch(
        "tools.query._expand_query",
        side_effect=lambda text, _config: {
            "expanded": text,
            "terms_added": [],
            "intent": None,
        },
    ), patch("tools.query._cross_schema_search", side_effect=search) as cross_search:
        rows, expansion = _search_query_windows("abcdefghij", service, 5)

    assert [row["id"] for row in rows] == ["obs::1", "obs::2", "obs::3"]
    assert expansion["base_window_count"] == 3
    assert cross_search.call_count == 3


class TestQueryVault:
    """Integration tests for vault semantic search."""

    def _index_test_files(self, mock_config):
        """Create and index test files."""
        from tools.memory import index_vault

        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / "auth-decisions.md").write_text(
            "---\ntype: note\nimportance: high\ntags:\n  - security\n  - work\n---\n"
            "# Authentication Decisions\n\nWe decided to use OAuth 2.0 with PKCE flow."
        )
        (notes_dir / "python-tips.md").write_text(
            "---\ntype: note\nimportance: medium\n---\n"
            "# Python Tips\n\nUse list comprehensions for cleaner code."
        )

        journal_dir = mock_config.vault_path / "journal" / "jarvis" / "2026" / "02"
        journal_dir.mkdir(parents=True, exist_ok=True)
        (journal_dir / "20260207-test-entry.md").write_text(
            "---\ntype: journal\nimportance: medium\n---\n"
            "# Test Journal Entry\n\nDiscussed authentication architecture today."
        )

        index_vault()

    def test_query_vault_basic(self, mock_config):
        self._index_test_files(mock_config)

        result = query_vault("authentication decisions")
        assert result["success"] is True
        assert result["query"] == "authentication decisions"
        assert len(result["results"]) > 0
        assert result["total_in_collection"] >= 3

        # Check result format
        first = result["results"][0]
        assert "rank" in first
        assert "path" in first
        assert "title" in first
        assert "type" in first
        assert "importance" in first
        assert "relevance" in first
        assert "preview" in first

        # Paths should NOT have vault:: prefix (stripped for display)
        for r in result["results"]:
            assert not r["path"].startswith("vault::")

    def test_query_vault_with_filter(self, mock_config):
        self._index_test_files(mock_config)

        result = query_vault("authentication", filter={"directory": "notes"})
        assert result["success"] is True
        for r in result["results"]:
            # All results should be from notes directory
            assert r["path"].startswith("notes/")

    def test_query_vault_empty(self, mock_config):
        # Don't index anything
        result = query_vault("anything")
        assert result["success"] is True
        assert result["results"] == []
        assert "No documents indexed" in result.get("message", "")

    def test_query_vault_n_results_cap(self, mock_config):
        self._index_test_files(mock_config)

        # Request more than 20 should be capped
        result = query_vault("test", n_results=50)
        assert result["success"] is True
        # Should not exceed 20 or total docs (whichever is smaller)
        assert len(result["results"]) <= 20

    def test_expansion_metadata_in_response(self, mock_config):
        """When query triggers expansion, response should include expansion info."""
        self._index_test_files(mock_config)

        result = query_vault("auth flow setup")
        assert result["success"] is True
        assert "expansion" in result
        assert len(result["expansion"]["terms_added"]) > 0

    def test_expansion_disabled_no_metadata(self, mock_config):
        """When query doesn't trigger expansion, no expansion field in response."""
        self._index_test_files(mock_config)

        result = query_vault("quantum entanglement")
        assert result["success"] is True
        # No matching synonyms/intents -> no expansion field
        assert "expansion" not in result

    def test_chunk_deduplication(self, mock_config):
        """Multiple chunks from same file should be deduped to best match."""
        from tools.memory import index_file

        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        # Create a multi-section file about auth
        content = (
            "## Authentication Overview\n\n"
            + "OAuth authentication is used for secure login. " * 30
            + "\n\n"
            "## Authorization Rules\n\n"
            + "Authorization controls what users can access. " * 30
            + "\n\n"
            "## Unrelated Section\n\n" + "This section is about cooking recipes. " * 30
        )
        (notes_dir / "auth-guide.md").write_text(content)
        index_file("notes/auth-guide.md")

        result = query_vault("authentication", n_results=5)
        assert result["success"] is True

        # Should only appear once despite multiple matching chunks
        paths = [r["path"] for r in result["results"]]
        assert paths.count("notes/auth-guide.md") == 1

    def test_importance_score_affects_relevance(self):
        """Documents with higher importance_score should get a relevance boost."""
        similarity = 1.0 - 0.5
        low_score = compute_unified_score(similarity, 0.3)
        high_score = compute_unified_score(similarity, 0.9)
        assert high_score > low_score

    def test_n_results_honored_when_reranking_applied(self, mock_config):
        """Defect #8 regression: callers asking for N results get N, even when
        reranking rescores the candidates (the old code returned top_k=10)."""
        mock_config.set(memory={"reranking": {"enabled": True}})
        self._index_test_files(mock_config)

        def fake_rerank(query, docs, vscores, cfg):
            return [s * 0.9 for s in vscores]  # new list → reranking_applied

        with patch("tools.reranking.rerank", side_effect=fake_rerank):
            result = query_vault("authentication decisions", n_results=2)

        assert result["success"] is True
        assert len(result["results"]) <= 2
        # Must be unconditional — without it, the count assertion also passes
        # when reranking silently fails to apply (vacuous for defect #8).
        assert result["reranking"]["applied"] is True

    def test_rerank_false_skips_an_enabled_reranker(self, mock_config):
        """Dedup asks for raw similarity on a 2s hook deadline: rerank=False
        must never reach the (host, up to 1.5s) cross-encoder."""
        mock_config.set(memory={"reranking": {"enabled": True}})
        self._index_test_files(mock_config)

        with patch("tools.reranking.rerank") as rerank, \
                patch("tools.reranking.rerank_multi") as rerank_multi:
            result = query_vault("authentication decisions", n_results=3, rerank=False)

        assert result["success"] is True
        assert result["results"]
        rerank.assert_not_called()
        rerank_multi.assert_not_called()
        assert all("similarity" in r for r in result["results"])


class TestDocRead:
    """Tests for document read by ID (renamed from memory_read)."""

    def _index_test_file(self, mock_config):
        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / "read-test.md").write_text("# Read Test\n\nContent for reading.")

        from tools.memory import index_file

        index_file("notes/read-test.md")

    def test_doc_read_basic(self, mock_config):
        self._index_test_file(mock_config)

        # Read using bare path (backward compatible)
        result = doc_read(["notes/read-test.md"])
        assert result["success"] is True
        assert len(result["documents"]) == 1
        assert result["documents"][0]["id"] == "vault::notes/read-test.md"
        assert result["documents"][0]["path"] == "notes/read-test.md"
        assert "Content for reading" in result["documents"][0]["document"]
        assert "metadata" in result["documents"][0]

    def test_doc_read_with_namespace(self, mock_config):
        """Should also work when called with full namespaced ID."""
        self._index_test_file(mock_config)

        result = doc_read(["vault::notes/read-test.md"])
        assert result["success"] is True
        assert len(result["documents"]) == 1
        # id is the raw namespaced ID, path is the display form
        assert result["documents"][0]["id"] == "vault::notes/read-test.md"
        assert result["documents"][0]["path"] == "notes/read-test.md"

    def test_doc_read_missing(self, mock_config):
        self._index_test_file(mock_config)

        result = doc_read(["notes/read-test.md", "notes/nonexistent.md"])
        assert result["success"] is True
        assert len(result["documents"]) == 1
        assert "notes/nonexistent.md" in result["not_found"]

    def test_doc_read_no_ids(self):
        result = doc_read([])
        assert result["success"] is False
        assert "No IDs" in result["error"]

    def test_backward_compat_alias(self):
        """memory_read should still work as alias for doc_read."""
        assert memory_read is doc_read


class TestCollectionStats:
    """Tests for collection statistics (renamed from memory_stats)."""

    def _index_test_files(self, mock_config):
        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / "stat-note-1.md").write_text(
            "---\ntype: note\n---\n# Note One\n\nFirst note."
        )
        (notes_dir / "stat-note-2.md").write_text(
            "---\ntype: idea\n---\n# Idea One\n\nAn idea."
        )

        from tools.memory import index_vault

        index_vault()

    def test_collection_stats_basic(self, mock_config):
        self._index_test_files(mock_config)

        result = collection_stats()
        assert result["success"] is True
        assert result["total_documents"] >= 2
        assert len(result["samples"]) > 0

        sample = result["samples"][0]
        assert "path" in sample
        assert "title" in sample
        assert "type" in sample
        # Paths should not have vault:: prefix
        assert not sample["path"].startswith("vault::")
        # Samples should have schema field
        assert "schema" in sample
        assert sample["schema"] in ("local", "obsidian")

    def test_collection_stats_detailed(self, mock_config):
        self._index_test_files(mock_config)

        result = collection_stats(detailed=True)
        assert result["success"] is True
        # New API uses category_breakdown and vault_type_breakdown
        # instead of type_breakdown and namespace_breakdown
        assert "core_documents" in result
        assert "vault_documents" in result
        assert result["vault_documents"] >= 2

    def test_collection_stats_empty(self, mock_config):
        result = collection_stats()
        assert result["success"] is True
        assert result["total_documents"] == 0
        assert result["samples"] == []
        assert "No documents indexed" in result.get("message", "")

    def test_backward_compat_alias(self):
        """memory_stats should still work as alias for collection_stats."""
        assert memory_stats is collection_stats


class TestCrossSchemaQuery:
    """Tests for cross-schema query results (local.memories + obsidian.documents)."""

    def test_query_includes_schema_field(self, mock_config):
        """Test that query results include schema field instead of tier."""
        from tools.memory import index_file

        test_file = mock_config.vault_path / "notes" / "test.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Test\nTest content")
        index_file("notes/test.md")

        # Query
        result = query_vault("test")
        assert result["success"]
        assert len(result["results"]) > 0

        # Check schema field (replaces old tier field)
        for res in result["results"]:
            assert "schema" in res
            assert res["schema"] == "obsidian"  # Vault files are in vault schema

    def test_query_includes_source_field(self, mock_config):
        """Test that query results include source field."""
        from tools.memory import index_file

        test_file = mock_config.vault_path / "notes" / "test.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Test\nTest content")
        index_file("notes/test.md")

        # Query
        result = query_vault("test")
        assert result["success"]

        # Check source field - vault documents have source="vault"
        for res in result["results"]:
            assert "source" in res
            assert res["source"] == "vault"

    def test_query_mixed_schema_results(self, mock_config):
        """Test query with both core and vault results."""
        # Index a vault file
        from tools.memory import index_file

        test_file = mock_config.vault_path / "notes" / "test.md"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text("# Test\nTest content for schema testing")
        index_file("notes/test.md")

        # Add a core observation
        from tools.content import content_write

        content_write(
            content="Core observation about schema testing content",
            content_type="observation",
            importance_score=0.8,
        )

        # Query
        result = query_vault("schema testing")
        assert result["success"]
        assert len(result["results"]) >= 2

        # Should have both schemas
        schemas = {res["schema"] for res in result["results"]}
        assert "obsidian" in schemas or "local" in schemas

    def test_query_increments_core_retrieval_count(self, mock_config):
        """Test that querying increments core retrieval counts."""
        # Add core observation
        from tools.content import content_write, content_read

        write_result = content_write(
            content="Test observation for retrieval count", content_type="observation"
        )
        doc_id = write_result["id"]

        # Initial count should be 0, reading increments it to 1
        read_result = content_read(doc_id)
        assert read_result["retrieval_count"] == 1.0

        # Query (should increment)
        query_vault("retrieval count")

        # Check count increased (read again increments, so should be >= 3)
        read_result2 = content_read(doc_id)
        assert read_result2["retrieval_count"] >= 3.0


class TestIncrementRetrievalCountsFractional:
    """Tests for fractional retrieval count increments."""

    def test_fractional_increment(self, mock_config):
        """increment=0.01 adds 0.01 to count."""
        from tools.content import content_write

        write_result = content_write(
            content="Test observation for fractional increment",
            content_type="observation",
        )
        doc_id = write_result["id"]

        _increment_retrieval_counts([doc_id], increment=0.01)

        row = mock_config.db.get_core(doc_id)
        assert row["retrieval_count"] == pytest.approx(0.01)

    def test_rounds_to_two_decimals(self, mock_config):
        """0.01 + 0.01 + 0.01 = 0.03 (no float noise)."""
        from tools.content import content_write

        write_result = content_write(
            content="Test rounding",
            content_type="observation",
        )
        doc_id = write_result["id"]

        for _ in range(3):
            _increment_retrieval_counts([doc_id], increment=0.01)

        row = mock_config.db.get_core(doc_id)
        assert row["retrieval_count"] == pytest.approx(0.03, abs=0.001)

    def test_default_increment_is_one(self, mock_config):
        """No increment arg -> adds 1.0 (backward compat)."""
        from tools.content import content_write

        write_result = content_write(
            content="Test default increment",
            content_type="observation",
        )
        doc_id = write_result["id"]

        _increment_retrieval_counts([doc_id])

        row = mock_config.db.get_core(doc_id)
        assert row["retrieval_count"] == pytest.approx(1.0)

    def test_skips_vault_ids(self, mock_config):
        """Vault (vault::) IDs are not incremented — only local.memories tracks counts."""
        from tools.memory import index_file

        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / "skip-test.md").write_text("# Skip Test\nContent")
        index_file("notes/skip-test.md")

        # Should not crash on vault:: IDs
        _increment_retrieval_counts(
            ["vault::notes/skip-test.md"], increment=0.01
        )

        # Vault rows don't have retrieval_count column
        row = mock_config.db.get_vault("vault::notes/skip-test.md")
        assert row is not None
        # vault rows don't track retrieval_count
        assert "retrieval_count" not in row or row.get("retrieval_count") is None


class TestSemanticContextDeduplication:
    """Redundant local memory/observation/worklog records inject once."""

    class FakeEmbeddingService:
        def __init__(self, vectors):
            self.vectors = vectors

        def encode_batch(self, texts, batch_size=8):
            return [self.vectors[text] for text in texts]

    @staticmethod
    def _entry(doc_id, doc_type, document, project="jarvis", relevance=0.9):
        return {
            "doc_id": doc_id,
            "parent_file": doc_id,
            "relevance": relevance,
            "similarity": relevance,
            "document": document,
            "metadata": {"type": doc_type, "project": project},
            "_schema": "local",
        }

    def test_prefers_durable_memory_over_redundant_transient_records(self):
        entries = [
            self._entry("worklog::1", "worklog", "worklog", relevance=0.95),
            self._entry("obs::1", "observation", "observation", relevance=0.93),
            self._entry("memory::global::1", "memory", "memory", relevance=0.90),
            self._entry("obs::2", "observation", "different", relevance=0.88),
        ]
        service = self.FakeEmbeddingService(
            {
                "worklog": [1.0, 0.0],
                "observation": [0.99, 0.01],
                "memory": [0.98, 0.02],
                "different": [0.0, 1.0],
            }
        )

        result, suppressed = _semantic_deduplicate_context(
            entries, service, similarity_threshold=0.90
        )

        assert {entry["doc_id"] for entry in result} == {
            "memory::global::1",
            "obs::2",
        }
        assert suppressed == 2

    def test_does_not_merge_equally_similar_records_across_projects(self):
        entries = [
            self._entry("memory::a::1", "memory", "first", project="a"),
            self._entry("obs::b", "observation", "second", project="b"),
        ]
        service = self.FakeEmbeddingService(
            {"first": [1.0, 0.0], "second": [1.0, 0.0]}
        )

        result, suppressed = _semantic_deduplicate_context(
            entries, service, similarity_threshold=0.90
        )

        assert result == entries
        assert suppressed == 0

    def test_embedding_failure_preserves_all_candidates(self):
        entries = [
            self._entry("obs::1", "observation", "one"),
            self._entry("worklog::1", "worklog", "two"),
        ]
        service = MagicMock()
        service.encode_batch.side_effect = RuntimeError("model unavailable")

        result, suppressed = _semantic_deduplicate_context(entries, service, 0.90)

        assert result == entries
        assert suppressed == 0


class TestSemanticContextFractionalBump:
    """Tests for fractional retrieval bumps in semantic_context()."""

    def test_bumps_fractionally(self, mock_config):
        """semantic_context() calls increment with configured value."""
        mock_config.set(
            memory={
                "context_enrichment": {"passive_retrieval_increment": 0.05},
            }
        )

        from tools.content import content_write

        write_result = content_write(
            content="Observation about career goals and strategic planning",
            content_type="observation",
            importance_score=0.8,
        )
        doc_id = write_result["id"]

        # Call semantic_context (should fractionally increment)
        semantic_context("career goals", threshold=0.0)

        # Check retrieval count was bumped (column-level, not metadata)
        row = mock_config.db.get_core(doc_id)
        count = row["retrieval_count"]
        assert count == pytest.approx(0.05, abs=0.001)

    def test_no_bump_when_zero(self, mock_config):
        """passive_retrieval_increment=0 -> no increment call."""
        mock_config.set(
            memory={
                "context_enrichment": {"passive_retrieval_increment": 0},
            }
        )

        from tools.content import content_write

        write_result = content_write(
            content="Observation about career goals zero increment",
            content_type="observation",
            importance_score=0.8,
        )
        doc_id = write_result["id"]

        semantic_context("career goals zero increment", threshold=0.0)

        row = mock_config.db.get_core(doc_id)
        assert row["retrieval_count"] == pytest.approx(0.0)

    def test_core_results_display_full_content(self, mock_config):
        """Core results in semantic_context should have display_mode='full'."""
        from tools.content import content_write

        content_write(
            content="Strategic observation about architecture decisions",
            content_type="observation",
            importance_score=0.9,
        )

        result = semantic_context("architecture decisions", threshold=0.0)
        assert len(result["matches"]) > 0

        # Core results should have full content display
        for match in result["matches"]:
            # match has 'type' from metadata — observations come from core
            if match["type"] == "observation":
                assert match["display_mode"] == "full"

    def test_vault_results_display_reference(self, mock_config):
        """Vault results in semantic_context should have display_mode='reference'."""
        from tools.memory import index_file

        notes_dir = mock_config.vault_path / "notes"
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / "ref-test.md").write_text(
            "# Reference Test\n\nContent about architecture."
        )
        index_file("notes/ref-test.md")

        result = semantic_context("architecture", threshold=0.0)
        assert len(result["matches"]) > 0

        # Vault results should have reference display
        for match in result["matches"]:
            if match["type"] in ("document", "vault", "note"):
                assert match["display_mode"] == "reference"

    def test_budget_tracking(self, mock_config):
        """semantic_context returns budget_used with core and vault breakdown."""
        from tools.content import content_write

        content_write(
            content="Budget test observation content",
            content_type="observation",
            importance_score=0.8,
        )

        result = semantic_context("budget test", threshold=0.0, budget=8000)

        assert "budget_used" in result
        assert "local" in result["budget_used"]
        assert "vault" in result["budget_used"]
        assert "total" in result["budget_used"]
        assert result["budget_used"]["total"] == 8000

    def test_large_entry_cross_bucket_spending(self, mock_config):
        """An entry larger than any single budget bucket is still selected
        when total remaining budget across all buckets is sufficient.

        Regression test for: entries just over budget/3 (e.g. 2687 chars with
        budget=8000 → bucket=2666) were silently dropped because _try_spend
        only checked individual buckets, not the combined total.
        """
        from tools.content import content_write

        # Budget=300 → 3-way split → 100 per bucket.
        # Write one local entry (small, to confirm budget flow works)
        content_write(
            content="Small local memory",
            content_type="observation",
            importance_score=0.8,
        )

        # Build a large document (150 chars > 100 per-bucket cap)
        large_content = "Security finding: " + "x" * 132  # exactly 150 chars

        # Mock _cross_schema_search to return a mix with a large remote entry
        from tools.schema_registry import SchemaEntry, SchemaKind
        large_remote_row = {
            "id": "obs::remote-large-001",
            "document": large_content,
            "metadata": {},
            "category": "observation",
            "scope": "global",
            "source": "auto-extract",
            "importance_score": 0.9,
            "retrieval_count": 0,
            "created_at": "2026-03-08T00:00:00+00:00",
            "distance": 0.1,  # very close = high similarity
            "_schema": "remote_test",
        }
        small_remote_row = {
            "id": "obs::remote-small-002",
            "document": "Short remote memory",
            "metadata": {},
            "category": "observation",
            "scope": "global",
            "source": "auto-extract",
            "importance_score": 0.7,
            "retrieval_count": 0,
            "created_at": "2026-03-08T00:00:00+00:00",
            "distance": 0.3,
            "_schema": "remote_test",
        }

        with patch("tools.query._cross_schema_search") as mock_search, \
             patch("tools.query._remote_count", return_value=2), \
             patch("tools.schema_registry.get_searchable_schemas") as mock_gss:
            # Registry returns remote schema so _remote_count works
            mock_gss.return_value = [
                SchemaEntry("remote_test", SchemaKind.REMOTE, "memories",
                            remote_name="test"),
            ]
            mock_search.return_value = [large_remote_row, small_remote_row]

            result = semantic_context(
                "security findings", threshold=0.0, budget=300
            )

        # The large entry (150 chars) exceeds any single bucket (100 chars)
        # but fits in total budget (300). It must be selected.
        ids = [m["id"] for m in result["matches"]]
        assert "remote-large-001" in ids, (
            f"Large remote entry not selected despite total budget being sufficient. "
            f"Selected: {ids}"
        )

    def test_entry_exceeding_total_budget_is_dropped(self, mock_config):
        """An entry larger than the total budget is correctly dropped."""
        from tools.content import content_write

        content_write(
            content="Anchor observation",
            content_type="observation",
            importance_score=0.5,
        )

        huge_content = "x" * 500  # 500 chars > total budget of 300

        with patch("tools.query._cross_schema_search") as mock_search, \
             patch("tools.query._remote_count", return_value=1), \
             patch("tools.schema_registry.get_searchable_schemas") as mock_gss:
            from tools.schema_registry import SchemaEntry, SchemaKind
            mock_gss.return_value = [
                SchemaEntry("remote_test", SchemaKind.REMOTE, "memories",
                            remote_name="test"),
            ]
            mock_search.return_value = [{
                "id": "obs::remote-huge-001",
                "document": huge_content,
                "metadata": {},
                "category": "observation",
                "scope": "global",
                "source": "auto-extract",
                "importance_score": 0.9,
                "retrieval_count": 0,
                "created_at": "2026-03-08T00:00:00+00:00",
                "distance": 0.1,
                "_schema": "remote_test",
            }]

            result = semantic_context(
                "huge entry test", threshold=0.0, budget=300
            )

        ids = [m["id"] for m in result["matches"]]
        assert "remote-huge-001" not in ids


class TestSemanticContextReranking:
    """Tests for cross-encoder reranking in semantic_context()."""

    @pytest.fixture(autouse=True)
    def _no_remote_schemas(self):
        """Isolate from remote schemas registered by other tests."""
        import tools.schema_registry as sr
        from tools.schema_registry import SchemaEntry, SchemaKind
        saved = sr._registry
        sr._registry = [e for e in sr._registry if e.kind != SchemaKind.REMOTE]
        yield
        sr._registry = saved

    def test_reranking_applied_to_semantic_context(self, mock_config):
        """When reranking is enabled, semantic_context calls rerank on deduped candidates."""
        import time as _time
        from tools.content import content_write

        mock_config.set(memory={"reranking": {"enabled": True}})
        content_write(
            content="First observation about database migrations",
            content_type="observation",
            importance_score=0.8,
        )
        _time.sleep(0.002)  # ensure distinct millisecond IDs
        content_write(
            content="Second observation about API design patterns",
            content_type="observation",
            importance_score=0.7,
        )

        with patch("tools.reranking.rerank") as mock_rerank:
            # Return identity (simulate graceful fallback) to not break the flow
            mock_rerank.side_effect = lambda q, docs, vscores, cfg: vscores

            # Raw cosine spans -1..1; -1 disables the quality gate for this
            # reranking integration test.
            result = semantic_context("database migrations", threshold=-1.0)

            # rerank should have been called with the query and documents
            assert mock_rerank.called, (
                f"rerank not called. matches={len(result.get('matches', []))}, "
                f"total_searched={result.get('total_searched')}"
            )
            call_args = mock_rerank.call_args
            assert call_args[0][0] == "database migrations"  # query
            assert len(call_args[0][1]) >= 2  # documents list
            assert len(call_args[0][2]) >= 2  # vector scores list

    def test_reranking_skipped_when_disabled(self, mock_config):
        """When reranking is disabled, semantic_context skips cross-encoder."""
        mock_config.set(memory={"reranking": {"enabled": False}})

        from tools.content import content_write

        content_write(
            content="Observation with reranking disabled",
            content_type="observation",
            importance_score=0.8,
        )

        with patch("tools.reranking.rerank") as mock_rerank:
            semantic_context("reranking disabled test", threshold=0.0)
            assert not mock_rerank.called

    def test_reranking_reorders_results(self, mock_config):
        """Cross-encoder reranking can reorder results by relevance."""
        import time as _time
        from tools.content import content_write

        mock_config.set(memory={"reranking": {"enabled": True}})
        content_write(
            content="Irrelevant observation about cooking recipes and food",
            content_type="observation",
            importance_score=0.9,
        )
        _time.sleep(0.002)  # ensure distinct millisecond IDs
        content_write(
            content="Highly relevant observation about SQL injection vulnerabilities",
            content_type="observation",
            importance_score=0.5,
        )

        def fake_rerank(query, docs, vscores, cfg):
            """Simulate cross-encoder that boosts the SQL injection doc."""
            alpha = cfg.get("alpha", 0.7)
            reranker_scores = []
            for doc in docs:
                if "SQL injection" in doc:
                    reranker_scores.append(0.95)
                else:
                    reranker_scores.append(0.1)
            # Min-max normalize
            min_s, max_s = min(reranker_scores), max(reranker_scores)
            if max_s - min_s > 1e-9:
                norm = [(s - min_s) / (max_s - min_s) for s in reranker_scores]
            else:
                norm = [0.5] * len(reranker_scores)
            return [alpha * n + (1 - alpha) * v for n, v in zip(norm, vscores)]

        with patch("tools.reranking.rerank", side_effect=fake_rerank):
            result = semantic_context("SQL injection vulnerabilities", threshold=0.0)

        matches = result["matches"]
        assert len(matches) >= 2
        # The SQL injection doc should rank first after reranking
        assert "SQL injection" in matches[0]["content"]


class TestBuildCoreFilterGeneric:
    """Test generic JSONB fallback in _build_core_filter."""

    def test_unknown_key_produces_jsonb_condition(self):
        """Unknown key falls back to metadata->>key = value."""
        conds, params = _build_core_filter({"workstream": "vuln-mgmt"})
        assert any("metadata->>%s = %s" in c for c in conds)
        assert "workstream" in params
        assert "vuln-mgmt" in params

    def test_multiple_unknown_keys(self):
        """Multiple unknown keys produce multiple JSONB conditions."""
        conds, params = _build_core_filter({
            "workstream": "vuln-mgmt",
            "git_branch": "main",
        })
        jsonb_conds = [c for c in conds if "metadata->>%s = %s" in c]
        assert len(jsonb_conds) == 2

    def test_known_keys_still_use_columns(self):
        """Known keys (type, importance, tags) use column conditions."""
        conds, params = _build_core_filter({
            "type": "observation",
            "workstream": "vuln-mgmt",
        })
        assert "category = %s" in conds
        jsonb_conds = [c for c in conds if "metadata->>%s = %s" in c]
        assert len(jsonb_conds) == 1

    def test_empty_values_skipped(self):
        """Empty/None values in filter dict are ignored."""
        conds, params = _build_core_filter({
            "workstream": "",
            "git_branch": None,
        })
        jsonb_conds = [c for c in conds if "metadata->>%s = %s" in c]
        assert len(jsonb_conds) == 0

    def test_mixed_known_and_unknown(self):
        """Mixed known + unknown keys produce correct conditions."""
        conds, params = _build_core_filter({
            "type": "observation",
            "importance": 0.7,
            "tags": "security",
            "workstream": "vuln-mgmt",
            "git_branch": "main",
        })
        assert "category = %s" in conds
        assert "importance_score >= %s" in conds
        jsonb_conds = [c for c in conds if "metadata->>%s = %s" in c]
        assert len(jsonb_conds) == 2

    def test_no_filter_dict(self):
        """None filter dict produces only status condition."""
        conds, params = _build_core_filter(None)
        assert conds == ["status = 'active'"]
        assert params == []


class TestBuildVaultFilterGeneric:
    """Test generic JSONB fallback in _build_vault_filter."""

    def test_unknown_key_produces_jsonb_condition(self):
        """Unknown key falls back to metadata->>key = value."""
        conds, params = _build_vault_filter({"author": "jarvis"})
        assert any("metadata->>%s = %s" in c for c in conds)
        assert "author" in params
        assert "jarvis" in params

    def test_known_vault_keys_use_columns(self):
        """Known vault keys (directory, type, etc.) use column conditions."""
        conds, params = _build_vault_filter({
            "directory": "notes",
            "author": "jarvis",
        })
        assert "directory = %s" in conds
        jsonb_conds = [c for c in conds if "metadata->>%s = %s" in c]
        assert len(jsonb_conds) == 1

    def test_empty_filter_dict(self):
        """Empty filter dict produces no conditions."""
        conds, params = _build_vault_filter({})
        assert conds == []
        assert params == []


class TestQueryVaultGenericFilter:
    """Integration tests for query_vault with generic metadata filters."""

    def test_filter_by_project(self, mock_config):
        """Filter by project narrows results."""
        from tools.content import content_write

        content_write(
            content="Security finding in framework",
            content_type="learning",
            extra_metadata={"scope": "project", "project": "personio-framework"},
        )
        content_write(
            content="Security finding in portal",
            content_type="pattern",
            name="portal-finding",
            extra_metadata={"scope": "project", "project": "developer-portal"},
        )

        result = query_vault(
            "security finding",
            filter={"project": "personio-framework"},
        )
        assert result["success"]
        for r in result["results"]:
            if r["schema"] == "local":
                assert r.get("id", "").startswith("learning::")

    def test_filter_by_session_id(self, mock_config):
        """Filter by session_id via generic JSONB filter."""
        from tools.content import content_write

        content_write(
            content="Session A observation",
            content_type="observation",
            session_id="session-aaa",
        )
        content_write(
            content="Session B observation",
            content_type="observation",
            session_id="session-bbb",
        )

        result = query_vault(
            "session observation",
            filter={"session_id": "session-aaa"},
        )
        assert result["success"]
