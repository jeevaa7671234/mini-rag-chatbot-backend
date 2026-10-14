# Run the assessment for free in VS Code (Windows)

This guide uses **local Ollama**, **local sentence-transformers**, and **embedded local Qdrant**. No paid account, API key, credit card, GPU, or Docker is needed. Internet access is needed for the initial software/model downloads. Once installed, the local inference/indexing workflow can run without a model API. CPU-only inference may be slow, especially on an older laptop.

## 1. Install the free prerequisites

- **Python 3.11, 64-bit** (recommended; Python 3.12 also supported). In a new PowerShell terminal, verify `py -3.11 --version`. Avoid accidentally using a different system Python.
- **Visual Studio Code** with the Microsoft Python extension.
- **Ollama for Windows** from <https://ollama.com/download/windows>. It usually runs in the background after installation.
- **Tesseract OCR for Windows** from <https://github.com/UB-Mannheim/tesseract/wiki>; add its folder (containing `tesseract.exe`) to the Windows `PATH`. Close and reopen the VS Code terminal. Test `tesseract --version`.

## 2. Extract ZIP, open folder and install dependencies

Extract the provided assessment ZIP. In VS Code select **File > Open Folder** and choose its inner `mini-rag-chatbot` directory (the directory containing `app/`, `docs/`, `requirements.txt`). Open **Terminal > New Terminal** (PowerShell), then run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If activation is blocked by PowerShell policy, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` then retry the activation command. In VS Code select **Python: Select Interpreter** and choose `.venv\Scripts\python.exe`.

The PyTorch/sentence-transformers installation is the largest Python download. Both CPU and GPU modes are possible; a GPU is not required.

## 3. Configure the model (exact tags matter)

```powershell
Copy-Item .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
notepad .env
ollama pull qwen2.5:3b
ollama list
```

In `.env`, replace `JWT_SECRET=change-this-to-a-long-random-secret` with the generated random secret. Confirm the following lines (already the defaults):

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
OLLAMA_MODEL=qwen2.5:3b
EMBEDDING_PROVIDER=sentence_transformers
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE=qdrant
QDRANT_PATH=data/qdrant
```

**The pulled Ollama tag MUST exactly equal `OLLAMA_MODEL`.** The local 3B model is about 1.9 GB on disk and supports tool calls, but tool decisions and policy-answer quality can be imperfect with a small model. If you have more available RAM (roughly 16 GB system RAM or more), you can optionally run `ollama pull qwen2.5:7b` and change `OLLAMA_MODEL=qwen2.5:7b`; do not change the embedding model. Re-indexing is **not necessary** for an LLM-only change.

If `ollama list` fails because the service is not running, launch the Ollama Windows app, or run `ollama serve` in another terminal when Ollama is not already running.

## 4. Preflight and ingestion (one-time)

```powershell
python -m scripts.check_environment
python -m app.ingestion
```

Expect: `Ingested 18 chunks with page and access metadata.` The first ingestion also downloads the free MiniLM embedding model, then creates the local Qdrant index in `data/qdrant`.

Keep the server **stopped** while re-ingesting; local Qdrant can refuse a second process opening the same storage directory.

## 5. Launch server and test with the original client

```powershell
python mint_tokens.py
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Leave the server terminal running. Open `test_client.html` in a browser, set `ws://localhost:8000/ws/chat`, and paste an **unexpired** JWT from `python mint_tokens.py`. Try:

1. As `emp-001` (HR level 1): **How many annual leave days do full-time employees receive?** (Should cite `leave_policy.pdf`.)
2. As `emp-001`: **What is the private performance rating distribution?** (Must not reveal manager-only policy.)
3. As `mgr-002` (HR level 2): **What are the five performance rating categories?** (Can use manager-only HR.)
4. As `exec-003`: **What is the APAC investment in the strategic plan?** (Should cite the scanned executive PDF.)
5. As `exec-003`: **What's the finance expense threshold?** (Must not reveal finance-only content despite level 3.)
6. As any user: **Who is my manager?** (Should invoke the mock context tool, taking about one second for the concurrent data fetches, in addition to model latency.)

A small local model can sometimes fail to request the optional tool; the preflight checks installation, but **only the actual demo confirms tool selection and answer quality**. Restart token generation if authentication expires (the test tokens live four hours).

## 6. Run automated tests (optional but recommended)

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Tests cover the WebSocket protocol, permissions, OCR, and tool timing. Passing mock tests does not prove live Ollama or Qdrant performance; the steps above test the real stack.

## Errors and fixes

| Symptom | Fix |
| --- | --- |
| `py -3.11` cannot find Python | Install Python 3.11 (64-bit) and reopen terminal. |
| `Activate.ps1` denied | Temporarily allow scripts in this PowerShell process as in Step 2. |
| `tesseract` not recognized / `TesseractNotFoundError` | Add Tesseract install folder to `PATH`, reopen VS Code terminal. |
| `No module named qdrant_client` or `sentence_transformers` | Confirm `.venv` is selected, then install `requirements.txt` in that environment. |
| Ollama `model not found` / 404 | `ollama list`; then pull the **same exact tag** as `OLLAMA_MODEL`. |
| Ollama connection refused | Start Ollama and verify `http://127.0.0.1:11434/api/tags` responds locally. |
| `No vector index found` | Run `python -m app.ingestion` before serving; verify `VECTOR_STORE=qdrant` is unchanged. |
| Qdrant file-lock error | Stop the server and any other processes using `data/qdrant`, then retry ingestion. |
| `auth_failed` | Regenerate JWTs after saving your current `JWT_SECRET`; paste a fresh token. |
| Answers are slow | CPU local model is inference-bound; a GPU or the optional free-tier Groq API can help. |
| Responses miss information | Try targeted questions and inspect OCR/chunks; increase local model size if hardware permits. |

## Optional hosted free tier (requires internet)

Groq supports `openai/gpt-oss-20b` for function calling on its free plan (subject to rate limits). To use it, create a Groq account/API key and switch the `.env` lines to:

```dotenv
LLM_PROVIDER=groq
GROQ_MODEL=openai/gpt-oss-20b
GROQ_API_KEY=your-own-key-here
```

In that mode Ollama does not need to run. Do not use the former free-tier Groq `llama-3.3-70b-versatile`: Groq retired it for free/developer accounts in August 2026. **Do not share or commit your `.env` or API key.** The base project still uses local MiniLM embeddings and Qdrant even when the LLM is remote.

Provider availability can change. This guide was checked against the provider model catalogs on **14 October 2026**; the ZIP's live hosted-model path has not been verified using your personal API key.
