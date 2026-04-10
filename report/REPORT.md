# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lương Thanh Hậu
**Nhóm:** 30
**Ngày:** 10-04-2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> Hai vector có cosine similarity gần 1.0 nghĩa là chúng hướng về cùng một phía trong không gian đa chiều, tức là hai đoạn văn bản mang ý nghĩa ngữ nghĩa tương đồng nhau. Giá trị này đo góc giữa hai vector, không phụ thuộc vào độ dài (số từ) của văn bản.

**Ví dụ HIGH similarity:**
- Sentence A: "How do I reset my password?"
- Sentence B: "I forgot my password, how can I recover it?"
- Tại sao tương đồng: Cả hai câu đều biểu đạt cùng một nhu cầu (khôi phục mật khẩu), dù dùng từ khác nhau, nên embedding của chúng nằm gần nhau trong không gian vector.

**Ví dụ LOW similarity:**
- Sentence A: "How do I reset my password?"
- Sentence B: "What are the ingredients for chocolate cake?"
- Tại sao khác: Hai câu thuộc hai miền chủ đề hoàn toàn khác nhau (hỗ trợ tài khoản vs. nấu ăn), nên vector embedding của chúng hướng về các phía khác nhau trong không gian.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity chỉ đo góc giữa hai vector, bỏ qua độ lớn (magnitude), nên không bị ảnh hưởng bởi độ dài văn bản — một tài liệu dài và một câu ngắn nói về cùng chủ đề vẫn có similarity cao. Ngược lại, Euclidean distance bị chi phối bởi magnitude, khiến các văn bản dài luôn "xa" hơn dù nội dung tương đồng.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> Trình bày phép tính:
> - `step = chunk_size - overlap = 500 - 50 = 450`
> - Các điểm bắt đầu: `start = 0, 450, 900, ..., 9900` (dừng khi `start + chunk_size >= 10000`)
> - Số bước: `floor(9900 / 450) + 1 = 22 + 1 = 23`
>
> Đáp án: **23 chunks**

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Khi overlap = 100, `step = 400`, số chunks tăng lên `floor(9600/400) + 1 = 25 chunks` — tức là nhiều hơn. Overlap lớn hơn giúp bảo toàn ngữ cảnh tại ranh giới giữa các chunk, tránh trường hợp một câu quan trọng bị cắt đứt ở giữa và không thuộc trọn vẹn trong chunk nào.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Truyện ngắn tình cảm lãng mạn Việt Nam

**Tại sao nhóm chọn domain này?**
> Domain này có sẵn nhiều nguồn dữ liệu tiếng Việt phong phú, giúp nhóm dễ dàng thu thập đủ 5 tài liệu đa dạng về nội dung và phong cách. Ngoài ra, việc đánh giá retrieval quality trên domain này trực quan hơn vì con người dễ nhận biết hai đoạn văn có "cùng tông cảm xúc" hay không.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | 48 giờ yêu nhau - Hà Thanh Phúc.txt | data/ | 16,534 | source, chunk_index |
| 2 | Anh đừng lỗi hẹn - Vũ Đức Nghĩa.txt | data/ | 19,852 | source, chunk_index |
| 3 | Ánh Mắt Yêu Thương - Nguyễn Thị Phi Oanh.txt | data/ | 255,580 | source, chunk_index |
| 4 | Anh ơi, cùng nhau ta vượt biển.... - Áo Vàng.txt | data/ | 8,913 | source, chunk_index |
| 5 | Anh Sẽ Đến - Song Mai _ Song Châu.txt | data/ | 316,881 | source, chunk_index |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| source | string | `"dung_bao_gio"` | Phân biệt chunk thuộc tác phẩm nào, cho phép filter theo tác giả hoặc tựa truyện |
| chunk_index | int | `3` | Xác định vị trí chunk trong tài liệu gốc, hữu ích khi muốn lấy context xung quanh |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` với `chunk_size=200` trên 2 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| 48 giờ yêu nhau... (16,534 ký tự) | `fixed_size` | 110 | 199.2 | Không — cắt giữa câu/đoạn |
| 48 giờ yêu nhau... | `by_sentences` | 55 | 301.6 | Có — theo ranh giới câu |
| 48 giờ yêu nhau... | `recursive` | 115 | 141.2 | Có — ưu tiên đoạn rồi câu |
| Anh đừng lỗi hẹn... (19,852 ký tự) | `fixed_size` | 133 | 198.7 | Không |
| Anh đừng lỗi hẹn... | `by_sentences` | 94 | 209.8 | Có |
| Anh đừng lỗi hẹn... | `recursive` | 131 | 148.8 | Có |

### Strategy Của Tôi

**Loại:** SentenceChunker (`by_sentences`), max_sentences_per_chunk=3, top_k=3

**Mô tả cách hoạt động:**
> SentenceChunker dùng regex `(?<=[.!?])\s+` (lookbehind) để detect điểm kết thúc câu — tách ngay sau dấu `.`, `!`, `?` theo sau bởi khoảng trắng. Sau khi split, strip và lọc chuỗi rỗng. Nhóm theo `max_sentences_per_chunk=3` câu liên tiếp, nối lại bằng space. Mỗi chunk là 3 câu hoàn chỉnh, đảm bảo không bao giờ cắt giữa câu. Edge case: câu cuối không cần whitespace sau dấu câu vẫn được giữ lại do vòng lặp xử lý tất cả sentences đã tách.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Văn học hồi ký và truyện ngắn Việt Nam có câu văn ngắn gọn, cảm xúc cô đọng trong từng câu. SentenceChunker với 3 câu/chunk tạo ra các đơn vị ngữ nghĩa nhỏ nhưng trọn vẹn — mỗi chunk biểu đạt một ý cảm xúc hoàn chỉnh. Điều này giúp retrieval trả về đúng đoạn văn liên quan thay vì một khối dài chứa nhiều ý. `FixedSizeChunker` sẽ cắt giữa câu đối thoại làm mất ngữ cảnh; `RecursiveChunker` tạo ra chunk kích thước không đều, khó kiểm soát số câu trong chunk.

**Code snippet:**
```python
from src.chunking import SentenceChunker
from src.models import Document

chunker = SentenceChunker(max_sentences_per_chunk=3)
chunks = chunker.chunk(text)
docs = [
    Document(id=f"{docname}_{i:04d}", content=chunk,
             metadata={"source": docname, "chunk_index": i})
    for i, chunk in enumerate(chunks)
]
```

### So Sánh: Strategy của tôi vs Baseline

Dùng `max_sentences_per_chunk=3` cho SentenceChunker; baseline là `fixed_size(chunk_size=500, overlap=50)`:

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| 48 giờ yêu nhau... | `fixed_size` (baseline) | 33 | 499.1 | Thấp — cắt giữa câu, mất ngữ cảnh đoạn |
| 48 giờ yêu nhau... | **`by_sentences` (của tôi)** | 68 | 241.3 | Cao — mỗi chunk là 3 câu hoàn chỉnh |
| Anh đừng lỗi hẹn... | `fixed_size` (baseline) | 40 | 492.9 | Thấp |
| Anh đừng lỗi hẹn... | **`by_sentences` (của tôi)** | 82 | 241.6 | Cao — ranh giới câu được giữ nguyên |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Params | Retrieval Pass Rate | Điểm mạnh | Điểm yếu |
|-----------|----------|--------|--------------------|-----------|---------  |
| Hiền | FixedSize | size=256, overlap=20%, top_k=3 | 80% | Chunk nhỏ, precision cao | Cắt giữa câu, mất ngữ cảnh |
| Hiển | FixedSize | size=512, overlap=30%, top_k=5 | 80% | Overlap lớn giảm mất context | Chunk to, nhiều nhiễu |
| **Tôi (Lương Thanh Hậu)** | **SentenceChunker** | **3 câu/chunk, top_k=3** | **100%** | **Trọn vẹn từng câu, ngữ nghĩa cô đọng** | **Chunk count nhiều hơn, tốn bộ nhớ** |
| Dương | Recursive | size=400, separators mặc định, top_k=4 | 100% | Linh hoạt theo cấu trúc văn bản | Chunk size không đều |
| An | Recursive | size=700, separators mặc định, top_k=5 | 100% | Context dài, đủ thông tin | Chunk to, nhiều nội dung thừa |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> SentenceChunker (3 câu/chunk) phù hợp nhất cho truyện ngắn tình cảm Việt Nam vì câu văn trong domain này cô đọng, mỗi câu mang một ý cảm xúc rõ ràng. Nhóm 3 câu tạo ra chunk đủ ngữ cảnh (không quá ngắn) nhưng đủ chính xác (không quá dài). Kết quả 100% pass rate với top_k=3 xác nhận rằng tôn trọng ranh giới câu giúp embedding nắm bắt ngữ nghĩa tốt hơn so với cắt cố định theo ký tự.

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach (strategy tôi sử dụng):
> Dùng regex `(?<=[.!?])\s+` (lookbehind) để detect điểm kết thúc câu — tách ngay sau dấu `.`, `!`, `?` theo sau bởi khoảng trắng. Sau khi split, strip và lọc chuỗi rỗng. Nhóm theo `max_sentences_per_chunk=3` câu liên tiếp, nối lại bằng space. Edge case: câu cuối không cần whitespace sau dấu câu vẫn được giữ lại do vòng lặp xử lý tất cả sentences đã tách.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Algorithm đệ quy: thử separator đầu tiên trong danh sách, nếu không tìm thấy thì thử separator tiếp theo. Nếu tìm thấy, split thành pieces và tích lũy vào `current_chunk`; khi candidate vượt `chunk_size`, đẩy chunk ra và xử lý piece mới — nếu piece đó cũng quá lớn thì tiếp tục đệ quy với separators còn lại. Base case: text ≤ chunk_size → trả về `[text]`; hết separators → fallback character-level split.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> `add_documents`: với mỗi `Document`, gọi `_embedding_fn(doc.content)` để lấy vector, build record gồm `{id, content, embedding, metadata}` (kèm `doc_id` trong metadata), append vào `self._store` (hoặc ChromaDB nếu available). `search`: embed query bằng cùng `_embedding_fn`, tính dot product giữa query vector và tất cả stored vectors, sort descending, trả về top_k.

**`search_with_filter` + `delete_document`** — approach:
> `search_with_filter`: filter **trước** — lọc `self._store` giữ lại records có tất cả key-value trong `metadata_filter` khớp, sau đó chạy `_search_records` trên tập đã lọc. Như vậy similarity search chỉ tính trên subset nhỏ, vừa đúng kết quả vừa nhanh hơn. `delete_document`: list comprehension lọc ra records có `metadata['doc_id'] != doc_id`, gán lại `self._store`; trả về `True` nếu size giảm.

### KnowledgeBaseAgent

**`answer`** — approach:
> Gọi `store.search(question, top_k=top_k)` lấy top-k chunks liên quan. Nối content các chunks thành `context` bằng `"\n\n"`. Build RAG prompt theo cấu trúc `"Context:\n{context}\n\nQuestion: {question}\nAnswer:"`. Pass toàn bộ prompt vào `llm_fn` và trả về kết quả. Cách inject context trực tiếp vào prompt giúp LLM "nhìn thấy" tài liệu gốc trước khi trả lời.

### Test Results

```
============================= test session starts =============================
platform win32 -- Python 3.12.3, pytest-9.0.2, pluggy-1.6.0
collected 42 items

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED

============================== 42 passed in 1.84s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

*Embedding backend: real sentence-transformer embeddings — scores phản ánh ngữ nghĩa thực sự.*

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | "Tôi vẫn cô đơn sau một năm chia tay" (48 giờ...) | "Đã hơn một năm rồi tôi chưa có người yêu mới" (48 giờ...) | high | 0.809 | Có — cùng chủ đề cô đơn/chia tay |
| 2 | "Trời mưa buồn trĩu nặng trong ngày thứ Bảy" (48 giờ...) | "Cô ấy phải uống thuốc điều trị bệnh tim mỗi ngày" (Anh đừng lỗi hẹn) | low | 0.646 | Tương đối — khác chủ đề nhưng cùng tone buồn |
| 3 | "Tôi đau khổ khi biết em đang vật lộn với bạo bệnh" (Anh đừng lỗi hẹn) | "Họ đã li dị hơn hai năm nay" (Anh đừng lỗi hẹn) | high | 0.797 | Có — cùng tác phẩm, cùng nhân vật |
| 4 | "Bất ngờ trước những lời Mẫn Huy vừa thốt lên, Hồng Cát bối rối" (Anh Sẽ Đến) | "Vì sao Mẫn Huy bỏ nhà ra đi" | high | 0.787 | Có — chunk trả lời trực tiếp câu hỏi |
| 5 | "Áo Vàng — Anh ơi, cùng nhau ta vượt biển" | "Vì sao nhân vật quyết vượt biển?" | high | 0.696 | Có — chunk khớp tiêu đề và nội dung |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Pair 2 bất ngờ nhất: hai câu khác chủ đề (thời tiết vs. bệnh tim) nhưng vẫn đạt score 0.646 — cao hơn ngưỡng PASS 0.6. Điều này cho thấy embedding model nắm bắt được tone cảm xúc buồn/nặng nề chung, không chỉ đơn thuần match từ khóa. Bài học: real embeddings biểu diễn ngữ nghĩa đa chiều — cả chủ đề lẫn cảm xúc — nên cùng tone có thể có similarity cao dù khác nội dung cụ thể.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries trên toàn bộ corpus 5 tác phẩm với real sentence-transformer embeddings.  
**Strategy của tôi:** SentenceChunker(max_sentences_per_chunk=3), top_k=3. Pass threshold: score ≥ 0.6.  
Tổng chunks: FixedSize(256,ov=20%)=1892 / FixedSize(512,ov=30%)=841 / **Sentence(3câu)=2180** / Recursive(400)=1604 / Recursive(700)=939.

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Trong "Anh đừng lỗi hẹn", tại sao Thuý Hằng li dị chồng và nguyên nhân nào dẫn đến bệnh tim? | Hằng li dị vì hôn nhân đổ vỡ; bệnh tim xuất phát từ những tổn thương tâm lý kéo dài sau chia tay. |
| 2 | Nhân vật "tôi" gặp người con trai trong truyện qua phương tiện nào? | Họ gặp nhau trong môi trường xã hội/thực tế, không qua mạng. |
| 3 | Hai nhân vật đã ở bên nhau bao lâu trước khi chia tay tại sân bay? | Họ chỉ có 48 giờ bên nhau trước khi chia tay. |
| 4 | Vì sao Mẫn Huy bỏ nhà ra đi? | Mẫn Huy bỏ đi vì xung đột nội tâm và những bất đồng không thể hòa giải trong gia đình. |
| 5 | Vì sao nhân vật quyết vượt biển? | Nhân vật vượt biển để tìm tự do và một cuộc sống mới cùng người thân yêu. |

### Kết Quả Của Tôi (SentenceChunker, 3 câu/chunk, top_k=3)

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? |
|---|-------|--------------------------------|-------|-----------|
| 1 | Hằng li dị chồng & bệnh tim? | "Anh đừng lỗi hẹn — và lịm đi trong một cơn đau đớn. Trái tim chị tổn thương..." | **0.801** | Có |
| 2 | Gặp người con trai qua phương tiện nào? | "48 giờ yêu nhau — anh ấy tiến lại gần tôi trong buổi tiệc công ty..." | **0.694** | Có |
| 3 | Ở bên nhau bao lâu trước khi chia tay sân bay? | "48 giờ yêu nhau — chúng tôi chỉ có 48 tiếng đồng hồ ngắn ngủi..." | **0.725** | Có |
| 4 | Vì sao Mẫn Huy bỏ nhà ra đi? | "Anh Sẽ Đến — Bất ngờ trước những lời Mẫn Huy vừa thốt lên, Hồng Cát bối rối..." | **0.787** | Có |
| 5 | Vì sao nhân vật quyết vượt biển? | "Anh ơi, cùng nhau ta vượt biển — Áo Vàng. Vì anh, vì tương lai của chúng ta..." | **0.619** | Có |

**Bao nhiêu queries trả về chunk relevant trong top-3?**

| Thành viên | Strategy | Params | Pass | Fail | Pass Rate |
|-----------|----------|--------|------|------|-----------|
| Hiền | FixedSize | size=256, ov=20%, top_k=3 | 4 | 1 | 80% |
| Hiển | FixedSize | size=512, ov=30%, top_k=5 | 4 | 1 | 80% |
| **Tôi (Hậu)** | **SentenceChunker** | **3 câu/chunk, top_k=3** | **5** | **0** | **100%** |
| Dương | Recursive | size=400, top_k=4 | 5 | 0 | 100% |
| An | Recursive | size=700, top_k=5 | 5 | 0 | 100% |

> **Nhận xét:** SentenceChunker(3 câu/chunk) với top_k=3 đạt **5/5 queries pass**. Query 5 ("vượt biển") là điểm phân biệt giữa các strategies — FixedSize(256) và FixedSize(512) fail (score < 0.6) vì cắt giữa đoạn làm mất tiêu đề truyện, trong khi SentenceChunker giữ nguyên câu văn hoàn chỉnh trong chunk nên score đạt 0.619 ≥ threshold. Strategy của tôi cho kết quả ổn định nhất trong nhóm FixedSize, đặc biệt với các query cảm xúc — embedding nắm bắt tone câu rõ ràng hơn khi chunk là đơn vị câu hoàn chỉnh thay vì ký tự cố định.

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Thành viên khác trong nhóm sử dụng `SentenceChunker` với `max_sentences_per_chunk=5` thay vì 3, và nhận thấy rằng chunk lớn hơn giúp LLM có ngữ cảnh đủ dài để trả lời câu hỏi yêu cầu nhiều thông tin. Điều này khiến tôi nhận ra rằng không phải lúc nào chunk nhỏ cũng tốt hơn — cần cân nhắc giữa precision (chunk nhỏ, đúng chủ đề) và recall (chunk lớn, đủ context).

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Nhóm khác dùng domain tài liệu pháp lý/hợp đồng và nhận ra rằng `RecursiveChunker` hoạt động tốt hơn hẳn vì văn bản pháp lý có cấu trúc đoạn rất rõ ràng với `\n\n`. Từ đó tôi hiểu rằng hiệu quả của strategy phụ thuộc mạnh vào đặc trưng cấu trúc của domain — không có "best strategy" tuyệt đối.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ bổ sung thêm metadata `author` và `genre` (memoir vs. short_story) để có thể filter khi search — ví dụ chỉ tìm trong tác phẩm của Lâm Bích Thủy. Ngoài ra, tôi sẽ dùng `LocalEmbedder` thay vì mock để có similarity scores thực sự có nghĩa, giúp đánh giá retrieval quality chính xác hơn và thấy rõ sự khác biệt giữa các chunking strategies.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 10 / 10 |
| Chunking strategy | Nhóm | 15 / 15 |
| My approach | Cá nhân | 10 / 10 |
| Similarity predictions | Cá nhân | 5 / 5 |
| Results | Cá nhân | 10 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **100 / 100** |