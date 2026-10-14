# Mini RAG Chatbot Backend

FastAPI · WebSocket token streaming · Qdrant retrieval-time RBAC · local semantic embeddings · scanned PDF OCR · concurrent employee-context tool.

**[Start here: README_SUBMISSION.md](README_SUBMISSION.md)** — installation, Windows/Linux setup, ingestion, server/client testing, RBAC rationale, async design, verification matrix, and limitations.

All **seven original Acme Corp assessment PDFs** are included under `docs/`. The unchanged employer-provided browser client is `test_client.html`, and the task statement is archived in `ASSIGNMENT_README.md`.

Run: `python -m app.ingestion` → `python mint_tokens.py` → `uvicorn app.main:app --reload` → open `test_client.html`.

Run offline tests: `python -m pip install -r requirements-test.txt` → `python -m pytest -q`.

**Windows/VS Code zero-cost setup:** [RUN_IN_VSCODE_WINDOWS.md](RUN_IN_VSCODE_WINDOWS.md). Run `python -m scripts.check_environment` after installing the real dependencies and pulling the model.
