# Báo cáo Day 7 — Agentic RAG

- **Họ và tên:** Đỗ Lý Hoàng Nam
- **MSSV:** 2A202603777
- **Lớp:** L3B

---

## 1. Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Tiếp nối trực tiếp corpus từ Day 6: 2 báo cáo **World Bank Taking Stock** về kinh tế Việt Nam (`taking_stock_2023_08.pdf` — 08/2023 và `taking_stock_2025_03.pdf` — 03/2025), giấy phép World Bank Open Knowledge (`CC BY 3.0 IGO`).
  - Tổng cộng **88 trang nội dung** (40 trang từ bản 08/2023 và 48 trang từ bản 03/2025, đã loại bìa, mục lục, trang chuyển chương, tài liệu tham khảo).
  - Sử dụng chiến lược chunking tốt nhất của Day 6 (**Strategy A**: `fixed_size=500` ký tự, `overlap=50` ký tự), tổng cộng **558 chunks** được export từ PostgreSQL sang [starter/data/converted/chunks.json](../../starter/data/converted/chunks.json).
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
  - Chấm thủ công theo rubric [BENCHMARK.md](../../BENCHMARK.md): Context (0/1/2), Grounded Answer (0/1/2), Citation Validity (citation có trong context và thực sự hỗ trợ khẳng định), Abstention (từ chối đúng khi thiếu bằng chứng, không suy đoán).

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
  - Agent `dev`: [selected_dev/traces.json](../../results/agent/selected_dev/traces.json), [selected_dev/reviews.json](../../results/agent/selected_dev/reviews.json), [selected_dev/agent_summary.json](../../results/agent/selected_dev/agent_summary.json)
  - Agent `test`: [selected_test/traces.json](../../results/agent/selected_test/traces.json), [selected_test/reviews.json](../../results/agent/selected_test/reviews.json), [selected_test/agent_summary.json](../../results/agent/selected_test/agent_summary.json)
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
- **Giới hạn:** Corpus phần cơ bản chỉ gồm text trích xuất từ 2 PDF (88 trang) của Day 6; kích thước tập `test` cơ bản là 7 câu (5 có đáp án + 2 thiếu bằng chứng), đạt mức tối thiểu nên độ biến thiên thống kê trên `test` còn cao.

---

## 4. Thử thách nâng cao (+15 điểm)

- **Tài liệu, nguồn, quyền sử dụng, số trang và phiên bản:**
  - **Tài liệu:** *Vietnam Country Climate and Development Report (CCDR)*, World Bank Group, ấn bản tháng 07/2022 (`vietnam_ccdr_full.pdf`).
  - **Nguồn tải chính thức:** https://documents1.worldbank.org/curated/en/099355507062229876/pdf/P17724103036ee07909701028a237b6d18b.pdf
  - **Quyền sử dụng:** World Bank Open Knowledge / Public Disclosure Authorized (`CC BY 3.0 IGO`, trang 2).
  - **Số trang:** **100 trang vật lý** (trang 1–100).
- **Cách xử lý text, bảng, ảnh, biểu đồ; phạm vi index và phần chưa xử lý:**
  - **Toàn văn 100 trang ([adv_ingest.py](../../src/adv_ingest.py)):**
    - Render tất cả 100 trang thành PNG 150 DPI (`pages/p001.png`–`p100.png`) qua `pypdfium2`.
    - **Text:** Trích xuất toàn văn bằng `pypdf`, chia chunk overlap (`size=900`, `overlap=150`) -> **373 text chunks** gắn đúng số trang gốc (`ADV:wb-vietnam-ccdr-2022:pXXX:tYY`).
    - **Bảng, biểu đồ, ảnh/sơ đồ/bản đồ:** Quét 58 trang có dấu hiệu trực quan bằng Gemini Vision (`gemini-3.5-flash-lite`) để trích xuất cấu trúc bảng Markdown, trục/chú giải/giá trị biểu đồ, nhãn bản đồ/sơ đồ, kèm tọa độ `bbox_pct` để crop ảnh từng vùng (`crops/pXXX_elYY.png`) -> **47 visual elements** (**15 bảng**, **16 biểu đồ**, **16 ảnh/sơ đồ/bản đồ**).
    - **Vector store:** Toàn bộ **420 records** (`373 text + 47 visual`) được embed bằng `gemini-embedding-001` (3072 chiều) và lưu vào bảng PostgreSQL `pgvector` `adv_chunks` + index BM25. Chi phí index toàn tài liệu: `\$0.025656` (vision) + `\$0.012749` (embedding) = **`\$0.0384`** (`623.46s`).
    - **Phần chưa xử lý & ảnh hưởng:** **3 trang cuối (`98, 99, 100`)** là trang trắng / bìa sau có `0` ký tự và không chứa hình phân tích (được ghi log riêng trong [ingest_stats.json](../../starter/data/advanced/ingest_stats.json); không ảnh hưởng tới nội dung báo cáo). Đã kiểm chứng đối chiếu nội dung trích xuất với ảnh trang gốc cho toàn bộ bảng/hình trong benchmark và trang bị thêm tool `inspect_page_visual(page, focus_question)` để agent kiểm tra trực tiếp ảnh PNG trang gốc khi trả lời.
- **Cách chia benchmark, schema evidence, model và cấu hình đã chốt trên `dev`:**
  - **Benchmark ([adv_cases.json](../../starter/data/advanced/adv_cases.json)):** tổng **48 câu** không trùng lặp ý/đáp án giữa `dev` và `test`:
    - **`dev` (18 câu):** 15 có đáp án (`3 table`, `3 image_diagram`, `3 chart`, `3 multi_hop`, `3 text`) + `3 unanswerable` (`au01`–`au03`).
    - **`test` (30 câu):** 24 có đáp án (`6 table`, `5 image_diagram`, `5 chart`, `4 multi_hop`, `4 text`) + `6 unanswerable` (`at25`–`at30`, kiểm chứng từ khóa xuất hiện 0 lần trên 100 trang).
    - Không dùng ảnh trang trí (chapter opener photos) làm câu hỏi ảnh/biểu đồ; toàn bộ nhóm `image_diagram` dùng sơ đồ khái niệm, bản đồ nguy cơ/tài nguyên, sơ đồ tài chính và ma trận ưu tiên.
  - **Thử nghiệm & chốt trên `dev` (`gemini-3.1-flash-lite`):**
    - `adv_text_baseline` (`text-only`, không `inspect_page_visual`, `budget=2500`): `13/15` câu có đáp án được trả lời (bị `abstain` ở `ad05` bản đồ Fig 9 và `ad07` treemap Fig 3), `3/3` từ chối đúng, trung bình `4.06` tool calls/câu, chi phí `\$0.03385`.
    - **`adv_multimodal_vision` (Đã chốt):** Hybrid RRF (Dense `adv_chunks` + BM25, `top_k=6`, `budget=4500`) + `inspect_page_visual` + `get_page_text`. Đạt **`18/18` (100%) trên `dev`** (`15/15` có đáp án đúng & đủ, `3/3` từ chối đúng, `0` lỗi, trung bình `1.89` tool calls/câu, chi phí `\$0.018508`).

### Kết quả chạy và chấm thủ công toàn bộ 30 câu `test` (`adv_multimodal_vision`)

| Nhóm test | Đạt / tổng | Tỷ lệ | Lỗi / nhận xét |
|---|---:|---:|---|
| Bảng (`table`) | 6 / 6 | 100.0% | Chính xác hoàn toàn Table 1, 3, 4, 5, Annex 1, Annex 3 CGE (`at01`–`at06`). |
| Ảnh, sơ đồ (`image_diagram`) | 4 / 5 | 80.0% | Đúng `at07`–`at10`; **`at11` KHÔNG ĐẠT** (nhầm bảng Resilient p.94 với Decarbonizing p.95). |
| Biểu đồ (`chart`) | 5 / 5 | 100.0% | Chính xác Fig 2, 6, 8, 13, 20 nhờ `inspect_page_visual` (`at12`–`at16`). |
| Multi-hop (`multi_hop`) | 4 / 4 | 100.0% | Tổng hợp & tính đúng phép tính xuyên trang (`at17`–`at20`). |
| Text (`text`) | 3 / 4 | 75.0% | Đúng `at21`, `at22`, `at24`; **`at23` KHÔNG ĐẠT** (đọc nhầm `younger` thành `poorer` và đảo ngược xác suất 37%). |
| Thiếu bằng chứng (`unanswerable`) | 6 / 6 | 100.0% | Từ chối đúng toàn bộ `at25`–`at30`, không suy đoán. |
| **Toàn bộ** | **28 / 30** | **93.33%** | **Đạt ngưỡng $\ge$ 90% (27/30); 0 lỗi hệ thống.** |

<details>
<summary>Chi tiết trace, citation và nhãn đối chiếu bản gốc cho toàn bộ 30 câu test (bấm để mở)</summary>

| ID | Nhóm | Tools & trang đã xem | Citations | Nhãn & đối chiếu bản gốc |
|---|---|---|---|---|
| `at01` | table | `search` (p.18) | `p018:v01` | **Đạt** — Đúng 4 cột Thailand & Vietnam (Table 1, p.18). |
| `at02` | table | `search`, `inspect_page_visual(46)` | `p046:v00` | **Đạt** — Đúng -134.9B vs -1.3B và chênh lệch 133.8B (Table 3, p.46). |
| `at03` | table | `search`, `inspect_page_visual(48)` | `p048:v00`, `PAGE:48` | **Đạt** — Đúng Energy efficiency +0.2%, NZP without reforms -73.6% (Table 4, p.48). |
| `at04` | table | `search`, `inspect_page_visual(52)` | `PAGE:52` | **Đạt** — Đúng Wind (offshore) +\$39B, Solar +429% (Table 5, p.52). |
| `at05` | table | `search` (p.85) | `p085:t02` | **Đạt** — Đúng 267.18 TWh, 84.78%, 15.22%, Our World in Data (Annex 1, p.85). |
| `at06` | table | `search` x3 (p.89) | `p089:t03` | **Đạt** — Đúng 4.24%, 9.36%, Agr 2.31%, Heat 2.92%, Flood 4.00% (Annex 3, p.89). |
| `at07` | image_diagram | `search`, `inspect_page_visual(38)` | `PAGE:38` | **Đạt** — Đúng 5 thành phố & legend Relative Flood Risk (Fig 11, p.38). |
| `at08` | image_diagram | `search`, `inspect_page_visual(54)` | `PAGE:54` | **Đạt** — Đúng Quang Ninh & 5 nhãn trên bản đồ (Fig 14, p.54). |
| `at09` | image_diagram | `search`, `inspect_page_visual(71)` | `p071:v00` | **Đạt** — Đúng Private \$184B (~3.4%), Public \$130B (~2.4%), External \$54B (~1%) (Fig 21, p.71). |
| `at10` | image_diagram | `search` (p.94) | `p094:t00`, `p094:v00` | **Đạt** — Liệt kê đủ 7 hành động High Urgency + High Impact (Annex 4, p.94). |
| `at11` | image_diagram | `search` (p.94, 95) | `p094:v00` | **Không đạt** — Trả lời nhầm theo bảng Resilient (p.94: 7 vs 4) thay vì Decarbonizing (p.95: 8 vs 5). |
| `at12` | chart | `search`, `inspect_page_visual(15)` | `PAGE:15`, `p015:v00` | **Đạt** — Đúng \$12,695 và điểm cắt T+54 (Fig 2, p.15). |
| `at13` | chart | `search`, `inspect_page_visual(23)` | `PAGE:23` | **Đạt** — Đúng 54% (11%, 35%, 38%, 16%) (Fig 6, p.23). |
| `at14` | chart | `search`, `inspect_page_visual(28)` | `p028:t01`, `PAGE:28` | **Đạt** — Đúng 9.8% vs 12.3% năm 2050, tách nhau ~2030 (Fig 8, p.28). |
| `at15` | chart | `search`, `inspect_page_visual(51)` | `PAGE:51` | **Đạt** — Đúng 922/550, 788/276, 812/26 TWh; CPS 246 vs ADS 63 MtCO2e (Fig 13, p.51). |
| `at16` | chart | `search`, `inspect_page_visual(66)` | `PAGE:66` | **Đạt** — Đúng Electricity +23.6%, Transport -6.3% (Fig 20, p.66). |
| `at17` | multi_hop | `search`, `inspect_page_visual(85)` x2 | `p085:t01`, `p085:t02` | **Đạt** — Đúng 364.43 -> 844 MtCO2e (BAU 927.9), tăng 479.57 MtCO2e (p.85). |
| `at18` | multi_hop | `search` x2, `inspect_page_visual(27)` | `PAGE:17`, `PAGE:28` | **Đạt** — Đúng \$837M, \$447M (p.17) và 12–14.5% GDP / \$400–\$523B (p.28). |
| `at19` | multi_hop | `search` (p.50, 52) | `p050:v00`, `p052:v00` | **Đạt** — Đúng 3 ý Box 6 (p.50) & Coal -\$19B (-72%), Offshore wind +\$39B (+402%) (p.52). |
| `at20` | multi_hop | `search` (p.59, 60) | `p059:t02`, `p060:t03`, `p060:t04` | **Đạt** — Đúng \$15.6B (p.59), 27.53M VND/ha & 10.97 MtCO2e/yr (p.60). |
| `at21` | text | `search`, `inspect_page_visual(31)` | `PAGE:31` | **Đạt** — Đúng 2030 (5.6%, 6%, 6.2%), 2050 (7.6%, 7.9%, 10.6%), PFES ~\$400M (p.31). |
| `at22` | text | `search` (p.47) | `PAGE:47` | **Đạt** — Đúng 5 nguồn PM2.5 (~35%, ~25%, 20%, 10%, ~7%) & nguyên nhân (p.47). |
| `at23` | text | `search` (p.64) | `p064:t01`, `p064:t02` | **Không đạt** — Nhầm `younger` thành `poorer` và đảo ngược "Kinh 37% less likely" thành "ethnic minority 37% more likely" (p.64). |
| `at24` | text | `search` (p.72) | `p072:t02` | **Đạt** — Đúng 72/85 và 74/85 tổ chức tín dụng (p.72). |
| `at25` | unanswerable | `search` x4 | `[]` | **Đạt** — Từ chối đúng về JETP \$15.5B (không có trong báo cáo 07/2022). |
| `at26` | unanswerable | `search` x1 | `[]` | **Đạt** — Từ chối đúng về số trạm sạc VinFast. |
| `at27` | unanswerable | `search` x1 | `[]` | **Đạt** — Từ chối đúng về giá mở cửa tín chỉ carbon HOSE 2025. |
| `at28` | unanswerable | `search` x5 | `[]` | **Đạt** — Từ chối đúng về công suất tái hóa khí LNG Thị Vải 2024. |
| `at29` | unanswerable | `search` x2 | `[]` | **Đạt** — Từ chối đúng về ngân sách trồng rừng ngập mặn Bến Tre 2023. |
| `at30` | unanswerable | `search` x2 | `[]` | **Đạt** — Từ chối đúng về công suất khai thác lithium & tái chế pin 2030. |

</details>

- **Trace, nhãn, thời gian, chi phí và lỗi hệ thống:**
  - Traces & nhãn lưu tại [multimodal_vision_test/traces.json](../../results/advanced/multimodal_vision_test/traces.json), [reviews.json](../../results/advanced/multimodal_vision_test/reviews.json) và [eval_summary.json](../../results/advanced/multimodal_vision_test/eval_summary.json).
  - **Thời gian & chi phí chạy `test` (30 câu):** `0` lỗi hệ thống, trung bình `1.9` tool calls/câu, median latency `11,106.3 ms` (~11.1s/câu do gọi Vision tool + pacing), tổng chi phí LLM `\$0.033346` + retriever `\$0.000102` = **`\$0.033448`**.
- **Giới hạn, cách tái lập và đường dẫn benchmark:**
  - **Giới hạn:** Ở câu `at11`, các bảng kéo dài nhiều trang (Annex 4, p.94–97) chưa tự động gộp thành một entity logic nên truy xuất trả `p094` trước `p095`; ở câu `at23`, LLM tự diễn giải ngược tỷ lệ tương đối ("less likely" vs "more likely"). Ngoài ra, miễn phí API Gemini giới hạn 15 RPM / 500 RPD nên phải phân tách model giữa bước ingest (`gemini-3.5-flash-lite`) và đánh giá (`gemini-3.1-flash-lite`).
  - **Đường dẫn tái lập:** Tải PDF từ URL World Bank ở trên về `starter/data/advanced/vietnam_ccdr_full.pdf`, chạy `python -m src.adv_ingest`, sau đó chạy `python -m src.adv_run_eval --split test --config multimodal_vision --model gemini-3.1-flash-lite` và `python -m src.adv_build_reviews`. Benchmark gốc: [adv_cases.json](../../starter/data/advanced/adv_cases.json).

