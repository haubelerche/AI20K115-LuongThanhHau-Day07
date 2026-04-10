from __future__ import annotations

import uuid
from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb

            # TODO: initialize chromadb client + collection
            # Khởi tạo ChromaDB client (lưu trữ trong bộ nhớ, không ghi ra file) và tạo một
            # collection mới với tên duy nhất (thêm chuỗi hex ngẫu nhiên để tránh xung đột khi chạy nhiều test song song).
            client = chromadb.EphemeralClient()
            unique_name = f"{collection_name}_{uuid.uuid4().hex[:8]}"
            self._collection = client.get_or_create_collection(unique_name)
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        # Tạo vector embedding từ nội dung tài liệu, kết hợp metadata gốc với doc_id,
        # trả về dict chuẩn hóa gồm: id, content, embedding và metadata.
        embedding = self._embedding_fn(doc.content)
        metadata = {**doc.metadata, "doc_id": doc.id}
        return {
            "id": doc.id,
            "content": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        # Tạo embedding cho câu truy vấn, tính điểm tương đồng (dot product) giữa query và từng record,
        # sắp xếp giảm dần theo điểm và trả về top_k kết quả tốt nhất.
        query_embedding = self._embedding_fn(query)
        scored = []
        for record in records:
            score = _dot(query_embedding, record["embedding"])
            scored.append({**record, "score": score})
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    def _batch_embed(self, texts: list[str]) -> list[list[float]]:
        """Use embed_batch if available, otherwise fall back to per-item calls."""
        if hasattr(self._embedding_fn, "embed_batch"):
            return self._embedding_fn.embed_batch(texts)
        return [self._embedding_fn(t) for t in texts]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        if not docs:
            return

        texts = [doc.content for doc in docs]
        embeddings = self._batch_embed(texts)

        if self._use_chroma and self._collection is not None:
            ids, documents, metas = [], [], []
            for doc, embedding in zip(docs, embeddings):
                unique_id = f"{doc.id}_{self._next_index}"
                self._next_index += 1
                ids.append(unique_id)
                documents.append(doc.content)
                metas.append({**doc.metadata, "doc_id": doc.id})
            # Add in a single Chroma call
            self._collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metas)
        else:
            for doc, embedding in zip(docs, embeddings):
                metadata = {**doc.metadata, "doc_id": doc.id}
                self._store.append({
                    "id": doc.id,
                    "content": doc.content,
                    "embedding": embedding,
                    "metadata": metadata,
                })

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        # Tạo embedding cho câu truy vấn, nếu đang dùng ChromaDB thì gọi API query của Chroma,
        # nếu không thì gọi _search_records để tìm kiếm trực tiếp trong bộ nhớ.
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n,
                include=["documents", "metadatas", "distances"],
            )
            output = []
            for i, doc_text in enumerate(results["documents"][0]):
                output.append({
                    "content": doc_text,
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                })
            return output
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        # Trả về tổng số chunk đang lưu trữ: dùng collection.count() nếu đang dùng ChromaDB,
        # hoặc len(self._store) nếu dùng bộ nhớ trong.
        if self._use_chroma and self._collection is not None:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        # Lọc trước các chunk theo điều kiện metadata (nếu có),
        # sau đó tìm kiếm tương đồng chỉ trong tập đã lọc để kết quả vừa đúng metadata vừa liên quan đến câu hỏi.
        if self._use_chroma and self._collection is not None:
            query_embedding = self._embedding_fn(query)
            n = min(top_k, self._collection.count())
            if n == 0:
                return []
            kwargs: dict[str, Any] = {
                "query_embeddings": [query_embedding],
                "n_results": n,
                "include": ["documents", "metadatas", "distances"],
            }
            if metadata_filter:
                kwargs["where"] = metadata_filter
            results = self._collection.query(**kwargs)
            output = []
            for i, doc_text in enumerate(results["documents"][0]):
                output.append({
                    "content": doc_text,
                    "metadata": results["metadatas"][0][i],
                    "score": 1.0 - results["distances"][0][i],
                })
            return output

        if metadata_filter is None:
            records = self._store
        else:
            records = [
                r for r in self._store
                if all(r["metadata"].get(k) == v for k, v in metadata_filter.items())
            ]
        return self._search_records(query, records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        # Xóa tất cả các chunk thuộc tài liệu có doc_id khớp:
        # với ChromaDB thì tìm các id liên quan rồi xóa, với bộ nhớ trong thì lọc bỏ các record đó.
        if self._use_chroma and self._collection is not None:
            results = self._collection.get(where={"doc_id": doc_id})
            if not results["ids"]:
                return False
            self._collection.delete(ids=results["ids"])
            return True

        before = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < before
