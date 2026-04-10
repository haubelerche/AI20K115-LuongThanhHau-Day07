"""
Tests for AI20K115-LuongThanhHau-Day07

Covers:
  - models.Document
  - chunking: FixedSizeChunker, SentenceChunker, RecursiveChunker,
              compute_similarity, ChunkingStrategyComparator
  - store.EmbeddingStore
  - agent.KnowledgeBaseAgent
"""

import math
import sys
import os

# Allow `import src.X` so relative imports inside src/ resolve correctly
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

from src.models import Document
from src.chunking import (
    FixedSizeChunker,
    SentenceChunker,
    RecursiveChunker,
    compute_similarity,
    ChunkingStrategyComparator,
)
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent


# ---------------------------------------------------------------------------
# models.Document
# ---------------------------------------------------------------------------

class TestDocument:
    def test_basic_fields(self):
        doc = Document(id="d1", content="Hello world")
        assert doc.id == "d1"
        assert doc.content == "Hello world"
        assert doc.metadata == {}

    def test_metadata_stored(self):
        doc = Document(id="d2", content="text", metadata={"source": "web"})
        assert doc.metadata["source"] == "web"

    def test_metadata_not_shared_between_instances(self):
        doc1 = Document(id="a", content="x")
        doc2 = Document(id="b", content="y")
        doc1.metadata["key"] = "val"
        assert "key" not in doc2.metadata


# ---------------------------------------------------------------------------
# chunking.FixedSizeChunker
# ---------------------------------------------------------------------------

class TestFixedSizeChunker:
    def test_empty_string(self):
        chunker = FixedSizeChunker(chunk_size=10)
        assert chunker.chunk("") == []

    def test_short_text_returned_as_single_chunk(self):
        chunker = FixedSizeChunker(chunk_size=100)
        text = "Short text"
        assert chunker.chunk(text) == [text]

    def test_exact_size(self):
        chunker = FixedSizeChunker(chunk_size=5, overlap=0)
        chunks = chunker.chunk("abcde")
        assert chunks == ["abcde"]

    def test_no_overlap(self):
        chunker = FixedSizeChunker(chunk_size=3, overlap=0)
        chunks = chunker.chunk("abcdef")
        # Each chunk is at most 3 chars; step = 3
        assert all(len(c) <= 3 for c in chunks)
        assert chunks[0] == "abc"
        assert chunks[1] == "def"

    def test_with_overlap(self):
        chunker = FixedSizeChunker(chunk_size=4, overlap=2)
        chunks = chunker.chunk("abcdefgh")
        # step = 4 - 2 = 2
        # chunks starting at 0, 2, 4, 6
        assert chunks[0] == "abcd"
        assert chunks[1] == "cdef"

    def test_all_chunks_within_size(self):
        chunker = FixedSizeChunker(chunk_size=10, overlap=3)
        text = "x" * 50
        chunks = chunker.chunk(text)
        assert all(len(c) <= 10 for c in chunks)
        assert len(chunks) > 1


# ---------------------------------------------------------------------------
# chunking.SentenceChunker
# ---------------------------------------------------------------------------

class TestSentenceChunker:
    def test_empty_string(self):
        chunker = SentenceChunker(max_sentences_per_chunk=3)
        assert chunker.chunk("") == []

    def test_single_sentence(self):
        chunker = SentenceChunker(max_sentences_per_chunk=3)
        chunks = chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert "Hello world" in chunks[0]

    def test_groups_sentences(self):
        chunker = SentenceChunker(max_sentences_per_chunk=2)
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunker.chunk(text)
        # 4 sentences, 2 per chunk → 2 chunks
        assert len(chunks) == 2

    def test_max_sentences_per_chunk_one(self):
        chunker = SentenceChunker(max_sentences_per_chunk=1)
        text = "First. Second. Third."
        chunks = chunker.chunk(text)
        assert len(chunks) == 3

    def test_strips_whitespace(self):
        chunker = SentenceChunker(max_sentences_per_chunk=3)
        chunks = chunker.chunk("  Hello world.  ")
        assert all(c == c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# chunking.RecursiveChunker
# ---------------------------------------------------------------------------

class TestRecursiveChunker:
    def test_empty_string(self):
        chunker = RecursiveChunker(chunk_size=100)
        assert chunker.chunk("") == []

    def test_short_text_returned_whole(self):
        chunker = RecursiveChunker(chunk_size=100)
        text = "Short text."
        result = chunker.chunk(text)
        assert result == [text]

    def test_splits_on_double_newline(self):
        chunker = RecursiveChunker(chunk_size=20)
        text = "Paragraph one.\n\nParagraph two."
        chunks = chunker.chunk(text)
        assert len(chunks) == 2

    def test_all_chunks_within_size(self):
        chunker = RecursiveChunker(chunk_size=20)
        text = "word " * 40
        chunks = chunker.chunk(text)
        assert all(len(c) <= 20 for c in chunks)

    def test_custom_separators(self):
        # "aaa" + "|" + "bbb" = 7 chars which fits chunk_size=10, so they merge into one chunk.
        # "ccc" goes into a second chunk. Result: 2 chunks.
        chunker = RecursiveChunker(separators=["|"], chunk_size=10)
        text = "aaa|bbb|ccc"
        chunks = chunker.chunk(text)
        assert len(chunks) == 2
        assert "aaa" in chunks[0]
        assert "ccc" in chunks[1]


# ---------------------------------------------------------------------------
# chunking.compute_similarity
# ---------------------------------------------------------------------------

class TestComputeSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(compute_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(compute_similarity(a, b)) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(compute_similarity(a, b) - (-1.0)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert compute_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert compute_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_range_is_minus_one_to_one(self):
        import random
        random.seed(42)
        a = [random.gauss(0, 1) for _ in range(16)]
        b = [random.gauss(0, 1) for _ in range(16)]
        sim = compute_similarity(a, b)
        assert -1.0 <= sim <= 1.0


# ---------------------------------------------------------------------------
# chunking.ChunkingStrategyComparator
# ---------------------------------------------------------------------------

class TestChunkingStrategyComparator:
    def test_returns_all_strategies(self):
        comparator = ChunkingStrategyComparator()
        text = "One sentence. Two sentence. Three sentence. Four sentence. Five sentence."
        # chunk_size must be > default overlap (50), use 200 (default)
        result = comparator.compare(text, chunk_size=200)
        assert "fixed_size" in result
        assert "by_sentences" in result
        assert "recursive" in result

    def test_each_strategy_has_required_keys(self):
        comparator = ChunkingStrategyComparator()
        text = "Hello world. This is a test."
        result = comparator.compare(text)
        for strategy in result.values():
            assert "count" in strategy
            assert "avg_length" in strategy
            assert "chunks" in strategy

    def test_count_matches_chunks_length(self):
        comparator = ChunkingStrategyComparator()
        text = "Sentence one. Sentence two. Sentence three."
        result = comparator.compare(text)
        for strategy in result.values():
            assert strategy["count"] == len(strategy["chunks"])


# ---------------------------------------------------------------------------
# store.EmbeddingStore
# ---------------------------------------------------------------------------

def _make_store() -> EmbeddingStore:
    """Return an in-memory store with deterministic mock embeddings."""
    from src.embeddings import MockEmbedder
    return EmbeddingStore(embedding_fn=MockEmbedder())


class TestEmbeddingStore:
    def test_starts_empty(self):
        store = _make_store()
        assert store.get_collection_size() == 0

    def test_add_documents_increases_size(self):
        store = _make_store()
        docs = [Document(id="1", content="foo"), Document(id="2", content="bar")]
        store.add_documents(docs)
        assert store.get_collection_size() == 2

    def test_add_empty_list_is_noop(self):
        store = _make_store()
        store.add_documents([])
        assert store.get_collection_size() == 0

    def test_search_returns_results(self):
        store = _make_store()
        docs = [
            Document(id="a", content="cats are fluffy animals"),
            Document(id="b", content="dogs love to fetch"),
            Document(id="c", content="birds can fly"),
        ]
        store.add_documents(docs)
        results = store.search("cats", top_k=2)
        assert len(results) == 2
        for r in results:
            assert "content" in r

    def test_search_top_k_limit(self):
        store = _make_store()
        docs = [Document(id=str(i), content=f"document {i}") for i in range(10)]
        store.add_documents(docs)
        results = store.search("document", top_k=3)
        assert len(results) <= 3

    def test_search_with_filter(self):
        store = _make_store()
        docs = [
            Document(id="x", content="python programming", metadata={"lang": "python"}),
            Document(id="y", content="java programming", metadata={"lang": "java"}),
        ]
        store.add_documents(docs)
        results = store.search_with_filter("programming", top_k=5, metadata_filter={"lang": "python"})
        for r in results:
            assert r["metadata"].get("lang") == "python"

    def test_delete_document(self):
        store = _make_store()
        docs = [
            Document(id="del1", content="to be deleted"),
            Document(id="keep1", content="to be kept"),
        ]
        store.add_documents(docs)
        removed = store.delete_document("del1")
        assert removed is True
        assert store.get_collection_size() == 1

    def test_delete_nonexistent_returns_false(self):
        store = _make_store()
        store.add_documents([Document(id="z", content="hello")])
        assert store.delete_document("does_not_exist") is False

    def test_metadata_stored_with_doc_id(self):
        store = _make_store()
        store.add_documents([Document(id="m1", content="meta test", metadata={"author": "Alice"})])
        results = store.search("meta test", top_k=1)
        assert results[0]["metadata"]["doc_id"] == "m1"
        assert results[0]["metadata"]["author"] == "Alice"


# ---------------------------------------------------------------------------
# agent.KnowledgeBaseAgent
# ---------------------------------------------------------------------------

class TestKnowledgeBaseAgent:
    def _make_agent(self, answers=None):
        """Helper: create an agent with a mock LLM that returns fixed answers."""
        store = _make_store()
        docs = [
            Document(id="1", content="Python is a programming language."),
            Document(id="2", content="Machine learning uses data to learn."),
            Document(id="3", content="Neural networks are inspired by the brain."),
        ]
        store.add_documents(docs)

        call_log = []

        def mock_llm(prompt: str) -> str:
            call_log.append(prompt)
            return "mock answer"

        agent = KnowledgeBaseAgent(store=store, llm_fn=mock_llm)
        return agent, call_log

    def test_answer_returns_string(self):
        agent, _ = self._make_agent()
        result = agent.answer("What is Python?")
        assert isinstance(result, str)

    def test_llm_called_with_prompt_containing_question(self):
        agent, call_log = self._make_agent()
        agent.answer("What is Python?")
        assert len(call_log) == 1
        assert "What is Python?" in call_log[0]

    def test_prompt_contains_context(self):
        agent, call_log = self._make_agent()
        agent.answer("Tell me about machine learning")
        prompt = call_log[0]
        assert "Context:" in prompt

    def test_top_k_limits_context_chunks(self):
        agent, call_log = self._make_agent()
        agent.answer("question", top_k=1)
        # The context should contain only 1 chunk worth of content
        prompt = call_log[0]
        # With top_k=1 there should be exactly one chunk in context (no double newline separating chunks)
        context_section = prompt.split("Question:")[0]
        assert context_section.count("\n\n") <= 1

    def test_store_and_llm_fn_stored(self):
        store = _make_store()
        fn = lambda p: "ok"
        agent = KnowledgeBaseAgent(store=store, llm_fn=fn)
        assert agent.store is store
        assert agent.llm_fn is fn
