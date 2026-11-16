# Hướng dẫn chạy code Day 7 (`src/`)

1. Khởi động PostgreSQL + `pgvector` từ Day 6 (`127.0.0.1:55433`, database `day6_lab`).
2. Sao chép `.env.example` thành `.env` và điền `DATABASE_URL`, `GEMINI_API_KEY`.
3. Cài thư viện:
   ```bash
   uv venv .venv
   uv pip install -r requirements.txt
   ```
4. Chạy unit test:
   ```bash
   .venv\Scripts\python.exe -m unittest discover -s tests -v
   ```
5. Export dữ liệu Day 6 và tạo bộ 24 câu benchmark:
   ```bash
   .venv\Scripts\python.exe -m src.export_day6
   ```
6. Chạy benchmark retrieval (TF-IDF, Dense pgvector, Hybrid RRF, Hybrid+Filter, Hybrid+Rerank):
   ```bash
   .venv\Scripts\python.exe -m src.retrieval_eval --split dev
   .venv\Scripts\python.exe -m src.retrieval_eval --split test --configs tfidf:tfidf:3:800:false dense:dense:3:800:false rerank_imp2:rerank:5:1800:true
   ```
7. Chạy Agent thật và tổng hợp kết quả:
   ```bash
   .venv\Scripts\python.exe -m src.run_agent_eval --split dev --config both
   .venv\Scripts\python.exe -m src.run_agent_eval --split test --config both
   .venv\Scripts\python.exe -m src.build_reviews
   ```
