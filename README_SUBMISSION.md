# Mini RAG Chatbot Backend - Submission

## Executive overview

This is a FastAPI WebSocket backend for answering company policy questions with **retrieval-time authorization**. It extracts page-linked content from normal and scanned PDFs, embeds chunks, stores vectors with required access metadata, retrieves only material visible to the JWT-authenticated user, allows the language model to invoke **one** employee-context tool, and streams actual model deltas over `/ws/chat`.

**Data availability notice:** The assignment instructions (`ASSIGNMENT_README.md`) were the **only** original file made available for this implementation. The seven employer PDFs, original `mint_tokens.py`, and original `test_client.html` were **not supplied**. This project includes newly written compatible replacements for the two helpers and an explicitly **fictional PDF generator** for local tests. Its synthetic policy text is not a claim about the employer's actual policies. For evaluation with real policy facts, place the original PDFs under `docs/` before ingestion. The PDF mappings and WebSocket wire protocol match the instructions.

## 1. Installation / prerequisites

- Python **3.11 or 3.12** recommended (3.10+ should work), `pip`, and an internet connection for the first package/model install.
- Install **Tesseract OCR** system executable and ensure `tesseract --version` works in your terminal. This is needed automatically for scanned `exec/strategic_plan.pdf`; the `pytesseract` Python package alone is not sufficient.
- Either [Ollama](https://ollama.com/) with a tool-capable model **or** a free Groq API key. You do not need both.
- Qdrant is **embedded/local** through `qdrant-client` (no Docker or external Qdrant server required).

### Windows / PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edit .env, especially JWT_SECRET.
tesseract --version
ollama pull qwen2.5:3b
```

Install Tesseract separately and add the folder containing `tesseract.exe` to PATH. Restart the shell after changing PATH. Run `ollama serve` if Ollama is not already running.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env and supply a long random JWT_SECRET.
# Ubuntu: sudo apt-get install tesseract-ocr
# macOS with Homebrew: brew install tesseract
ollama pull qwen2.5:3b
```

Set a different `JWT_SECRET` in `.env`, e.g. `python -c "import secrets; print(secrets.token_urlsafe(32))"`. Do **not** commit `.env` or real tokens.

### Choice of models and providers

| Purpose | Default choice | Why |
|---|---|---|
| LLM | Ollama OpenAI-compatible API, `qwen2.5:3b` | Zero cloud keys/cost, supports OpenAI-format tool calls and real output streaming |
| Optional hosted LLM | Groq `llama-3.3-70b-versatile` | Fast, free-tier-friendly, good function calling; set `LLM_PROVIDER=groq` and `GROQ_API_KEY` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional) | Local/free, compact, adequate semantic matching for a small policy corpus |
| Vector database | Qdrant local mode | Real vector index, built-in boolean payload filters, easy to run without Docker |

Both LLMs use the same `OpenAICompatibleLLM` interface, and the application code calls the abstract `select_tool` and `stream` methods. Changing `LLM_PROVIDER`/model does not affect chat orchestration. The embedding and vector-store adapters are also replaceable independently.

**Real local run:** `LLM_PROVIDER=ollama`, `EMBEDDING_PROVIDER=sentence_transformers`, `VECTOR_STORE=qdrant` (these are already the `.env.example` defaults). If using Groq, edit only `LLM_PROVIDER` and `GROQ_API_KEY` (the same real Qdrant and embedding options work).

**Strictly for automated tests / offline demo:** `LLM_PROVIDER=mock`, `EMBEDDING_PROVIDER=hashing`, `VECTOR_STORE=sqlite`. The mock is **not an LLM** and returns a visibly marked `[DEMO ONLY]` response; hashing is lexical (not semantic); SQLite uses SQL cosine distance with SQL `WHERE` ACL filtering, but is not an approximate-nearest-neighbor engine. Do not confuse this mode with validation of generative answer quality or with the production configuration.

## 2. How to ingest the documents

Copy the seven **real** PDFs into the exact paths:

```
docs/
  hr/leave_policy.pdf
  hr/code_of_conduct.pdf
  hr/performance_review.pdf
  finance/expense_policy.pdf
  finance/travel_reimbursement.pdf
  exec/compensation_committee.pdf
  exec/strategic_plan.pdf
```

Then run from the project root:

```bash
python -m app.ingestion
```

The script OCRs each page with less than 40 extractable characters using 2x-resolution Tesseract, splits pages into chunks, embeds in batches of 64, then replaces the existing vector collection. It prints the number of chunks ingested. The index persists under `data/qdrant/`. Stop the server before re-ingesting: local Qdrant uses an exclusive storage lock. Unknown PDFs fail ingestion **rather than being assigned an unsafe default permission**, and missing expected PDFs fail explicitly rather than creating an incomplete index.

**Only have the README?** For a self-contained evaluation of mechanics, you can instead create *clearly synthetic* fixtures with:

```bash
python scripts/create_demo_docs.py
python -m app.ingestion
```

The generator intentionally refuses to overwrite existing PDF files. The strategic-plan sample is a genuine **image-only PDF** so OCR is exercised (not copied text masquerading as a scan). Do not submit fixture data as the employer's original content.

## 3. Run server and test the exact WebSocket protocol

```bash
python mint_tokens.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `test_client.html` directly in a browser. Paste one printed token, leave `ws://localhost:8000/ws/chat`, click Connect, then ask a question. The client requires **no build step**. The provided original test client (once available) should also work because the message envelope is the specified one.

On a fresh connection the first client frame must be authentication:

```json
{"type":"auth","token":"<signed JWT from mint_tokens.py>"}
```

The server returns:

```json
{"type":"auth_success","user_id":"emp-001","department":"hr","level":1}
```

The client asks:

```json
{"type":"message","text":"How many leave days does the policy provide?"}
```

The real model returns multiple successive tokens/deltas and one terminal frame:

```json
{"type":"stream","text":"According to "}
{"type":"stream","text":"the leave policy "}
{"type":"stream","text":"[leave_policy.pdf p.1] ..."}
{"type":"done"}
```

Those stream texts are **illustrative protocol examples, not guaranteed output**: text contents and delta sizes depend on the model and actual PDFs. Invalid, expired, or missing JWTs get `auth_failed` followed by a WebSocket policy close (`1008`). LLM and tool errors send a user-facing `stream` error followed by `done` while leaving the socket open. Successive authenticated messages work without reconnecting. `GET /health` reports process health only (not index/provider readiness).

Useful access checks:

| User | HR public (L1) | HR managers (L2) | Finance public (L1) | Exec confidential (L3) |
|---|---|---|---|---|
| `emp-001`: hr, L1 | Yes | No | No | No |
| `mgr-002`: hr, L2 | Yes | Yes | No | No |
| `exec-003`: exec, L3 | Yes | Yes | No | Yes |

A future finance user at L1 would see finance L1 **plus HR L1**, and not executive content.

## 4. Chunking strategy and justification

The unit is a **PDF page**, preserving `page` and `source_file` in every chunk for citation and review. Text is normalized and divided into windows of **900 characters**, with approximately **150 characters of overlap**. The splitter prefers paragraph breaks, then sentence endings, line/word boundaries late in the window. Pages are never combined. This is deliberately char-based rather than tokenizer-specific, keeping ingestion model-independent and readable for small policy PDFs. Roughly 900 characters capture several related policy sentences without swallowing whole documents; 150 characters carry context over a section boundary. Overlap is approximate when avoiding half-words. Short pages produce one chunk. OCR text uses exactly the same chunker.

A production iteration should tune chunk sizes against real policies using grounded Q&A/recall tests and potentially split on section headings or tables; the chosen limits are an explained baseline, not claimed optimal.

## 5. Retrieval-time RBAC (central security decision)

Each chunk includes exactly the required data: `department` from folder, integer `access_level` from the explicit seven-file allowlist, `source_file`, and 1-based `page` (plus `text` for answering).

Access is defined as:

```
(access_level <= authenticated_user.level)
AND
(department == authenticated_user.department OR department == 'hr')
```

In `app/store.py`, the **Qdrant search call** passes a `models.Filter` with a mandatory range condition and nested OR department condition in the `query_filter=` parameter to `query_points(...)`. Filtering happens **inside Qdrant before similarity ranking and `limit`**, not by trimming `top_k` results afterward. The alternative SQLite test store enforces the same predicate in SQL `WHERE` before similarity sort/limit. This avoids irrelevant high-similarity restricted text entering the prompt and also avoids missing an authorized lower-ranked document due to post-filtering.

The JWT's `sub`, department, and level are validated server-side using HS256 signature and mandatory timestamps. WebSocket JSON cannot override them. A requested employee-context tool call is accepted **only for the authenticated JWT subject**; it cannot read another employee even if an LLM or prompt injection supplies a different ID.

## 6. Async requests, real streaming, and the parallel tool

The LLM sees one JSON-schema function: `get_employee_context(user_id)`. Tool choice occurs via an asynchronous, short **non-streaming** model call with that schema; if selected, the application invokes the tool exactly once and streams the actual final answer via a second model call. This separates tool selection from user-facing token streaming, and works consistently across both Groq and Ollama. It costs an additional model round trip (an explicit tradeoff).

Within the tool, `asyncio.gather(_profile(user_id), _manager(user_id), _team(user_id))` starts all three mock operations together. Each sleeps for one second; the entire aggregation takes approximately **one second**, not three. The mocked response contains Jane Doe / Senior, John Smith, and Platform / size 8. It **does not** contain any current leave balance, and the system prompt explicitly forbids making one up.

WebSocket handling, network calls (`httpx.AsyncClient`), and tool calls are async. The synchronous sentence-transformer embed computation, SQLite operations, local Qdrant storage, and PDF processing are offloaded to worker threads or run in the separate ingestion CLI. **No synchronous I/O is performed in the WebSocket's `async def` event loop.** Streaming forwards provider SSE `delta.content` pieces as they arrive; it does not chunk a complete response to fake interactivity.

## 7. One improvement with more time

Add **evaluation-driven retrieval** using a labelled set of policy questions, section-aware/table-aware extraction, and a local reranker with measured retrieval recall and answer-grounding checks. The source PDFs were unavailable for evaluation in this implementation, so no measured retrieval accuracy is claimed.

## 8. One deliberate tradeoff

**Local, small, free dependencies over highly scalable operations.** Embedded Qdrant + a 384D CPU model gives reviewers a realistic filtered vector store without Docker, a database account, or embedding API costs. The compromise is model-download/setup overhead, limited concurrency for local Qdrant, and lower semantic recall than larger models or hybrid search. A second tradeoff is a two-pass model invocation for reliable portable tool calling (see §6).

## Test coverage, reproducibility, and limitations

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests exercise JWT success/failure, socket authentication order, WebSocket streaming/continued connection, lexical-vector SQL **query-time** RBAC for HR L1/L2, finance L1 and exec L3, forbidden cross-user tool calls, genuine image-only PDF OCR, PDF ACL validation, chunking, tool parallel elapsed time, and mock HTTP SSE chunk forwarding. They **do not use paid services** or require real policy documents. A full live Groq/Ollama and embedded-Qdrant smoke run with the employer's PDFs should also be performed before submitting to the recruiter; it was not possible to verify external LLM availability or the missing original policy corpus here.

The project is intentionally **small scale**. Not implemented: conversation memory across connections, tenant separation, automated credential rotation, rate limiting, systematic answer evaluation, or multi-worker vector-store operations. These would be necessary for a real enterprise rollout. For actual production use, deploy Qdrant as a service instead of local mode, harden audit/auth, redact private data from logs, and use secure WebSockets (`wss://`).

### Suggested demo questions

- As `emp-001`: “What is the leave policy?” then “What happens in managers' performance calibration?” The second must **not** cite private manager text.
- As `mgr-002`: “Describe the performance review process.”
- As `exec-003`: “What does the scanned strategic plan say about Redwood?” (the **synthetic** scan exercises OCR).
- As `exec-003`: “What is the finance expense policy?” The system must **not** reveal finance content despite the user's high level, because department restrictions still apply.
- As any authenticated user: “Who is my manager?” This exercises the single tool's concurrent mock calls.

## Project layout

```
app/
  main.py          # WS frames, JWT handshake, streaming, errors
  security.py      # signed JWT verification / role claims
  chat.py          # grounded prompt, filtered retrieval, model tool flow
  llm.py           # Groq/Ollama async provider + offline test mock
  store.py         # Qdrant filtered vectors / SQL filtered test backend
  embeddings.py    # local sentence-transformers / offline lexical hashes
  ingestion.py     # ACL map, PDF text+OCR, chunking, embedding CLI
  tools.py         # asyncio.gather employee context tool
scripts/create_demo_docs.py  # explicitly fictional, includes image-only scan
mint_tokens.py              # signed short-lived test user tokens
test_client.html            # no-build browser client
README_SUBMISSION.md        # this document
ASSIGNMENT_README.md        # uploaded instruction text
requirements*.txt
.env.example
.gitignore
```

### Development commits

See the Git commit history included with the submission. This is **real local Git history** for the implementation and validation stages, not a claim of multiple days' work. The original assignment README and synthetic helper are disclosed rather than presented as supplied company files.
