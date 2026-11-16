# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Đỗ Lý Hoàng Nam
- **MSSV:** 2A202603777
- **Lớp:** L3B

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tiếp nối trực tiếp corpus từ Day 6: 2 báo cáo **World Bank Taking Stock** về kinh tế Việt Nam (`taking_stock_2023_08.pdf` — 08/2023 và `taking_stock_2025_03.pdf` — 03/2025), giấy phép World Bank Open Knowledge (`CC BY 3.0 IGO`).
  - Tổng cộng **88 trang nội dung** (40 trang từ bản 08/2023 và 48 trang từ bản 03/2025, đã loại bìa, mục lục, trang chuyển chương, tài liệu tham khảo).
  - Sử dụng chiến lược chunking tốt nhất của Day 6 (**Strategy A**: `fixed_size=500` ký tự, `overlap=50` ký tự), tổng cộng **558 chunks** được export từ PostgreSQL sang [starter/data/converted/chunks.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/starter/data/converted/chunks.json).
- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** PostgreSQL 16 + `pgvector 0.8.7` (`day6_lab` trên `127.0.0.1:55433`, bảng `chunks`, exact cosine distance `<=>`).
  - **Embedding model:** `gemini-embedding-001` (`3072` chiều; `RETRIEVAL_DOCUMENT` khi index ở Day 6, `RETRIEVAL_QUERY` khi truy vấn ở Day 7).
- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent & Rerank model:** `gemini-3.5-flash-lite` qua `google-genai` SDK (native function calling, `temperature=0.0`, HTTP timeout 30s, tự động retry khi gặp `429`/`503`).
  - **Phân chia dev/test (24 câu):**
    - `dev` (17 câu): 15 câu có đáp án (`q01`–`q15` từ Day 6) + 2 câu thiếu bằng chứng (`u01`: hệ số Gini 2024; `u02`: thuế suất TNDN ưu đãi cho nhà đầu tư bán dẫn 2025).
    - `test` (7 câu): 5 câu có đáp án (`q16`–`q20` từ Day 6) + 2 câu thiếu bằng chứng (`u03`: lượng hành khách sân bay Long Thành Q1/2025; `u04`: thâm hụt thương mại song phương với Hàn Quốc Q4/2024).
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - Câu có đáp án có `gold` gồm `{source_id, page, evidence}` đối chiếu từ PDF gốc; `is_gold_hit` chuẩn hóa khoảng trắng (`_norm`) để không bị sai lệch do dấu xuống dòng PDF.
  - Câu thiếu bằng chứng được kiểm chứng trên toàn bộ 558 chunks (các từ khóa `gini`, `semiconductor`, `long thanh`, `south korea` đều xuất hiện **0/558** chunks).
  - Chấm thủ công theo rubric [BENCHMARK.md](file:///c:/Users/admin/LabTeam/TH%20day%2007/BENCHMARK.md): Context (0/1/2), Grounded Answer (0/1/2), Citation Validity (citation có trong context và thực sự hỗ trợ khẳng định), Abstention (từ chối đúng khi thiếu bằng chứng, không suy đoán).

---

## 2. Retrieval và thử nghiệm

### Bảng so sánh các phương pháp và cấu hình trên `dev` và `test`

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline (`tfidf`) | TF-IDF lexical, `top_k=3`, không filter, không rerank, `budget=800` chars | Baseline lexical offline | dev | 6/15 (40.0%) | 11/15 (73.3%) | 6/15 (40.0%) | 1.73 ms / \$0 |
| Phương pháp 2 (`dense`) | Dense `pgvector` Day 6 (`gemini-embedding-001`), `top_k=3`, `budget=800` chars | Truy xuất ngữ nghĩa dùng vector store Day 6 | dev | 4/15 (26.7%) | 8/15 (53.3%) | 4/15 (26.7%) | 44.39 ms (cached) / \$0.000061 (lần đầu) |
| Phương pháp 3 (`hybrid`) | Hybrid RRF (Dense + BM25, `rrf_k=60`), `top_k=3`, `budget=800` chars | Kết hợp khớp từ khóa chính xác (BM25) và ngữ nghĩa (Dense) | dev | 6/15 (40.0%) | 11/15 (73.3%) | 6/15 (40.0%) | 38.07 ms / \$0 |
| Cải thiện 1 (`hybrid_imp1`) | Hybrid RRF + `auto_filter` năm báo cáo + `top_k=5`, `budget=1800` chars | Chunk dài ~500 ký tự nên `budget=800` chỉ chứa được 1 chunk; tăng `budget=1800` chứa được 3+ chunk và `auto_filter` tránh nhiễu giữa báo cáo 2023/2025 | dev | 7/15 (46.7%) | 11/15 (73.3%) | 11/15 (73.3%) | 41.81 ms / \$0 |
| **Cải thiện 2 (`rerank_imp2` — Đã chọn)** | Hybrid RRF (`cand=10`) + Gemini Rerank + `auto_filter` + `top_k=5`, `budget=1800` chars | Dùng `gemini-3.5-flash-lite` chấm điểm lại top-10 candidate để đẩy đúng chunk chứa evidence lên rank 1–3 | dev | **11/15 (73.3%)** | **13/15 (86.7%)** | **13/15 (86.7%)** | 1398.94 ms / \$0.006154 |
| Baseline (`tfidf`) | TF-IDF lexical, `top_k=3`, `budget=800` chars | Đánh giá độc lập sau khi chốt cấu hình trên `dev` | test | 3/5 (60.0%) | 3/5 (60.0%) | 3/5 (60.0%) | 2.65 ms / \$0 |
| Baseline (`dense`) | Dense `pgvector` Day 6, `top_k=3`, `budget=800` chars | Đánh giá độc lập sau khi chốt cấu hình trên `dev` | test | 2/5 (40.0%) | 2/5 (40.0%) | 2/5 (40.0%) | 466.47 ms / \$0.000024 |
| **Cấu hình đã chọn (`rerank_imp2`)** | Hybrid RRF + Gemini Rerank + `auto_filter`, `top_k=5`, `budget=1800` chars | Đánh giá độc lập sau khi chốt cấu hình trên `dev` | test | **3/5 (60.0%)** | **4/5 (80.0%)** | **4/5 (80.0%)** | 1734.56 ms / \$0.001982 |

*(Chi phí embedding chỉ tính ở lần gọi đầu tiên chưa có trong `.cache/query_embeddings.json`; các lần chạy lại đọc cache có chi phí embedding = \$0.)*

### Phân tích 2 lỗi trên `dev`, nguyên nhân, thử nghiệm và lý do chọn cấu hình

1. **Lỗi 1 — Nghẽn `budget=800` ký tự ở Baseline (`dense_dev` và `tfidf_dev`):**
   - **Hiện tượng:** Trên `dev`, `dense` đạt `Hit@3 = 8/15` nhưng `Context Evidence Rate` tụt xuống chỉ còn `4/15` (`q07`, `q11`, `q12`, `q14` có evidence ở rank 2–3 nhưng bị rớt khỏi context).
   - **Nguyên nhân:** Mỗi chunk Strategy A dài ~500 ký tự cộng header `[chunk_id | source | page]` (~65 ký tự) nên 2 chunk cần >1100 ký tự; với `budget=800`, hàm `pack_context` chỉ nhét vừa **1 chunk duy nhất**.
   - **Thử nghiệm & kết quả:** Tăng `top_k=5`, `budget=1800` ở `hybrid_imp1` giúp `Context Evidence Rate` tăng vọt từ `6/15 (40.0%)` lên `11/15 (73.3%)`.

2. **Lỗi 2 — Agent từ chối nhầm & lỗi `503 ServerError` ở lần chạy `agent_selected` đầu tiên trên `dev`:**
   - **Hiện tượng:** Ở lần chạy đầu trên `dev`, `agent_selected` chỉ trả lời được 6/15 câu và gặp 2 lỗi `503 ServerError` (`q04`, `u02`), dù `rerank_imp2` đã đưa đúng evidence vào top-3 ở 13/15 câu.
   - **Nguyên nhân (đọc trace `q01`, `q03`, `q04`, `q06`):**
     - (a) Model xuất citation dạng gộp `[id1, id2]` trong khi bộ kiểm tra yêu cầu từng ngoặc riêng `[id1]` nên câu trả lời đúng bị hạ xuống `abstain`.
     - (b) Retry chỉ bắt `errors.ClientError` (`429`) mà bỏ sót `errors.ServerError` (`503`).
   - **Thử nghiệm & kết quả:** Sửa prompt + regex kiểm tra citation trong ngoặc `[...]` và bắt `errors.APIError` (`429` và `503`). Kết quả chạy lại `agent_selected` trên `dev`: **0 lỗi**, **15/15** câu có đáp án được trả lời đúng (mean `2.0/2.0`), **2/2** câu thiếu bằng chứng được từ chối.

3. **Lý do chọn cấu hình:** Theo tiêu chí chốt trước trên `dev` (ưu tiên `Context Evidence Rate`, `Grounded Answer`, `Abstention`, trong ngưỡng latency < 5s và chi phí < \$0.02), chọn **`rerank_imp2` + `agent_selected`** (`Hybrid RRF + Gemini Rerank + get_adjacent_chunks`, `max_searches=4`, `top_k=5`, `budget=2200`).

---

## 3. Agent và đánh giá

### Thiết kế Agent
- **Tools (`src/agent.py`):**
  1. `search_documents(query, year_filter, top_k)`: gọi retriever (`dense` cho Baseline; `rerank` cho Selected), hỗ trợ suy luận `year_filter` (`2023`/`2025`) từ câu hỏi khi người dùng nêu rõ ấn bản.
  2. `get_adjacent_chunks(chunk_id)`: lấy chunk liền kề (`window=1`) trên cùng tài liệu (chỉ bật ở `agent_selected`), giúp khắc phục các câu mà bằng chứng bị cắt ngang ranh giới chunk 500 ký tự (như `q08`, `q15` trên `dev` và `q20` trên `test`).
- **Điều kiện dừng & Context:** Dừng khi model không gọi thêm tool và trả JSON `{status, answer, citations}`, hoặc chạm giới hạn `max_searches` (`2` cho Baseline, `4` cho Selected). Context được deduplicate theo `chunk_id` và đóng gói theo `budget` (`800` vs `2200` ký tự). Nếu citation rỗng hoặc không nằm trong context, hệ thống ép chuyển về `abstain`.
- **Xử lý lỗi:** Tự động retry tối đa 5 lần với `429`/`503`; nếu vẫn lỗi, ghi nhận `status: "error"` riêng biệt, không thay bằng kết quả giả.

### Bảng tổng hợp Agent (`starter/agent_metrics.py`)

| Cấu hình / split | Tool use | Số lượt gọi TB | Grounded answer (0 / 1 / 2 và TB) | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| `agent_baseline` / `dev` | 17/17 (100%) | 1.65 | 9 / 0 / 6 — **0.80 / 2.00** | 6/6 (100%) | 6/6 (100%) | 2/2 (100%) | 9/15 (60.0%) |
| `agent_selected` / `dev` | 17/17 (100%) | 1.18 | 0 / 0 / 15 — **2.00 / 2.00** | 29/29 (100%) | 15/15 (100%) | 2/2 (100%) | 0/15 (0.0%) |
| `agent_baseline` / `test` | 7/7 (100%) | 1.71 | 2 / 1 / 2 — **1.00 / 2.00** | 3/3 (100%) | 3/3 (100%) | 2/2 (100%) | 2/5 (40.0%) |
| **`agent_selected` / `test`** | **7/7 (100%)** | **1.29** | **0 / 0 / 5 — 2.00 / 2.00** | **10/10 (100%)** | **5/5 (100%)** | **2/2 (100%)** | **0/5 (0.0%)** |

### Chấm Context 0/1/2 trên 5 câu `test` có đáp án
- **`agent_baseline` (`test`):** `0` = 2 câu (`q16`, `q20`), `1` = 1 câu (`q18` — chỉ có `p051:c002` về quy định pháp lý, thiếu `p051:c000`/`p051:c001` về thiếu phương pháp phân tích chi phí-lợi ích xã hội), `2` = 2 câu (`q17`, `q19`).
- **`agent_selected` (`test`):** `0` = 0 câu, `1` = 0 câu, `2` = **5 câu** (`q16`, `q17`, `q18`, `q19`, `q20` đều có đủ bằng chứng và nguồn/trang).

### Chi tiết từng câu trên tập `test` (`agent_selected`)

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| `q16` | Có đáp án | 1 lượt `search_documents` (`year_filter=2023`, `top_k=5`) -> lấy được `p011:c000`, `p019:c000` | Global GDP giảm từ 3.1% (2022) xuống 2.1% (2023); global trade giảm từ 6% xuống 1.7% `[A:wb-taking-stock-2023-08:p011:c000] [A:wb-taking-stock-2023-08:p019:c000]` | **Answer: 2, Context: 2, Cites: [True, True]** — Đúng, đủ số liệu, 2 citation hỗ trợ trực tiếp. |
| `q17` | Có đáp án | 1 lượt `search_documents` (`year_filter=2023`) -> `p042:c000` | Chỉ 77% ngân sách được giải ngân (chênh lệch 23%); chuẩn mực quốc tế < 5% `[A:wb-taking-stock-2023-08:p042:c000]` | **Answer: 2, Context: 2, Cites: [True]** — Chính xác tuyệt đối theo p.42. |
| `q18` | Có đáp án | 1 lượt `search_documents` -> `p051:c000`, `p051:c001`, `p051:c002` | Thiếu phương pháp hệ thống như phân tích chi phí-lợi ích xã hội (SCBA) và thiếu quy định đánh giá sớm `[A:wb-taking-stock-2023-08:p051:c000] [A:wb-taking-stock-2023-08:p051:c001] [A:wb-taking-stock-2023-08:p051:c002]` | **Answer: 2, Context: 2, Cites: [True, True, True]** — Đủ cả ý phương pháp lẫn bối cảnh quy định (khắc phục lỗi thiếu ý của Baseline). |
| `q19` | Có đáp án | 1 lượt `search_documents` (`year_filter=2025`) -> `p011:c001`, `p019:c002` | Tiêu dùng cá nhân đạt 54% GDP so với trung vị khu vực 61.7% do tiết kiệm hộ gia đình tăng trong bối cảnh bất định `[A:wb-taking-stock-2025-03:p011:c001] [A:wb-taking-stock-2025-03:p019:c002]` | **Answer: 2, Context: 2, Cites: [True, True]** — Đúng, đủ nguyên nhân và tỷ lệ so sánh. |
| `q20` | Có đáp án | 1 lượt `search_documents` -> `p058:c000` + `p058:c001` | 5 bộ (MoT, MoIT, MoF, MPI, MoST) và EVN (Viet Nam Electricity) `[A:wb-taking-stock-2025-03:p058:c000] [A:wb-taking-stock-2025-03:p058:c001]` | **Answer: 2, Context: 2, Cites: [True, True]** — Lấy được cả 2 chunk nối tiếp dù evidence gốc trải trên 2 chunk. |
| `u03` | Thiếu bằng chứng | 2 lượt `search_documents` ("Long Thanh International Airport passenger volume Q1 2025", "Long Thanh airport") -> không có dữ liệu liên quan | Từ chối: tài liệu Taking Stock không chứa thông tin lượng hành khách sân bay Long Thành Q1/2025. `citations: []` | **Abstention: True** — Tìm kiếm 2 lượt trước khi từ chối, không bịa đặt. |
| `u04` | Thiếu bằng chứng | 2 lượt `search_documents` ("bilateral trade deficit South Korea Q4 2024", "trade balance South Korea Vietnam 2024") -> không có dữ liệu | Từ chối: tài liệu Taking Stock không chứa thông tin thâm hụt thương mại song phương với Hàn Quốc Q4/2024. `citations: []` | **Abstention: True** — Tìm kiếm 2 lượt và từ chối chính xác. |

### Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí
- **Traces & nhãn:**
  - Retrieval: `results/retrieval/<config>_<split>/traces.json` & `summary.json`
  - Agent `dev`: [selected_dev/traces.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_dev/traces.json), [selected_dev/reviews.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_dev/reviews.json), [selected_dev/agent_summary.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_dev/agent_summary.json)
  - Agent `test`: [selected_test/traces.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_test/traces.json), [selected_test/reviews.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_test/reviews.json), [selected_test/agent_summary.json](file:///c:/Users/admin/LabTeam/TH%20day%2007/results/agent/selected_test/agent_summary.json)
- **Lệnh chạy tái lập:**
  - Unit tests: `.venv\Scripts\python.exe -m unittest discover -s tests -v` (20/20 pass)
  - Export Day 6: `.venv\Scripts\python.exe -m src.export_day6`
  - Retrieval benchmark: `.venv\Scripts\python.exe -m src.retrieval_eval --split dev` và `--split test`
  - Agent benchmark: `.venv\Scripts\python.exe -m src.run_agent_eval --split dev --config both` và `--split test --config both`
  - Tổng hợp agent: `.venv\Scripts\python.exe -m src.build_reviews`
- **Thời gian & chi phí Agent:**
  - `agent_baseline` (`test`, 7 câu): median `3080.3 ms`, LLM cost `\$0.002875`, retriever cost `\$0.000028`.
  - `agent_selected` (`test`, 7 câu): median `4936.9 ms`, LLM cost `\$0.003668`, retriever cost `\$0.003614` (tổng ~\$0.0073 cho 7 câu).

### Đánh đổi và giới hạn
- **Đánh đổi:** `agent_selected` tăng chất lượng trả lời từ `1.00/2.00` lên `2.00/2.00` (giảm từ chối nhầm từ 40% xuống 0%) nhưng đánh đổi độ trễ từ ~3.1s lên ~4.9s/câu và cần thêm chi phí rerank (~10 lần gọi/câu ở mức candidate). Khi dùng Gemini Free Tier (15 RPM), việc gọi tuần tự cả rerank và agent dễ chạm ngưỡng `429`, cần cơ chế retry/pacing.
- **Giới hạn:** Corpus chỉ gồm text trích xuất từ 2 PDF (88 trang) của Day 6 (chưa OCR biểu đồ/ảnh riêng biệt); kích thước tập `test` là 7 câu (5 có đáp án + 2 thiếu bằng chứng), đạt mức tối thiểu của phần cơ bản nên độ biến thiên thống kê trên `test` còn cao.
