# Mini RAG Chatbot Backend — Submission

**FastAPI · WebSocket streaming · Qdrant · Local embeddings · PDF/OCR · JWT/RBAC · concurrent tools**

## At a glance

This is a deliberately small RAG backend for the **seven Acme Corp policy PDFs supplied with the assignment**. It reads both normal and image-only PDF pages; creates page-aware chunks with explicit permissions; indexes embeddings in Qdrant; retrieves under **database-enforced access controls**; allows an LLM to request one authenticated employee-context function; and streams actual model response deltas to the **unchanged employer-provided `test_client.html`**.

The supplied original documents are already in `docs/`. The original task is preserved as [`ASSIGNMENT_README.md`](ASSIGNMENT_README.md). No PDF generation or replacement is needed to evaluate the submission.

### Trust boundary / architecture

```text
Input PDFs in docs/{hr,finance,exec}/
      │ native text (PyMuPDF) or automatic 3x OCR (Tesseract)
      ▼
Page-bounded normalized text → 900-character chunks, 150-character overlap
      │ allowlisted department + integer access level + source_file + page
      ▼
Local sentence-transformer embeddings → persistent Qdrant collection
                                                     ▲
WebSocket client ── JWT auth ── validated role ── Qdrant FILTERED similarity query
      │                                              │
      │             ONLY permitted policy text ───────┘
      ▼
OpenAI-compatible LLM adapter (Ollama or Groq)
      ├─ optional tool call: get_employee_context(authenticated_user_id)
      │       └─ asyncio.gather(profile, manager, team) ~1 sec total
      └─ genuine provider delta stream → {"type":"stream","text":"..."} → {"type":"done"}
```

## 1. Install and run ingestion

**Recommended:** Python 3.11 or 3.12, an internet connection for the initial model/package installation, and Tesseract OCR available on `PATH`. First run of sentence-transformers downloads a small pretrained model. No paid API or Docker is required with the default setup.

### Windows (PowerShell)

From the project root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Install [Tesseract OCR for Windows](https://github.com/UB-Mannheim/tesseract/wiki) separately, add the directory containing `tesseract.exe` to your Windows `PATH`, then start a **new** PowerShell terminal and verify `tesseract --version`.

Generate a fresh random JWT secret (`python -c "import secrets; print(secrets.token_urlsafe(32))"`) and paste it as `JWT_SECRET` in `.env`.

The default LLM uses free local Ollama. Install [Ollama](https://ollama.com/), then:

```powershell
ollama pull qwen2.5:3b
# Make sure Ollama is running (or use `ollama serve` if needed)
python -m app.ingestion
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
# Replace JWT_SECRET in .env with a random string.
# Linux Ubuntu/Debian: sudo apt-get install tesseract-ocr
# macOS with Homebrew: brew install tesseract
ollama pull qwen2.5:3b
python -m app.ingestion
```

Running the ingestion command should print `Ingested 18 chunks with page and access metadata.` for these original PDFs. It extracts all seven originals, including the genuinely image-only `docs/exec/strategic_plan.pdf`. The resulting local Qdrant index is persisted under `data/qdrant/`. Keep the server stopped when rebuilding this local index to avoid local Qdrant file locks.

**LLM provider choices:** Default is `LLM_PROVIDER=ollama`, `OLLAMA_BASE_URL=http://127.0.0.1:11434/v1`, `OLLAMA_MODEL=qwen2.5:3b` (no API credentials). Alternatively, set `LLM_PROVIDER=groq` and `GROQ_API_KEY=...` in `.env` for Groq's hosted free-tier API. The configured Groq model is `llama-3.3-70b-versatile`; change `GROQ_MODEL` if your account's supported models differ. The model interface is provider-independent (`ChatLLM` with `select_tool` / `stream`). Embeddings use `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Qdrant runs in local embedded mode, so a separate vector DB server is not required.

**Never commit** `.env`, real API tokens, or the local vector index (`data/` is excluded from Git).

## 2. Run the server

```bash
python mint_tokens.py
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

`mint_tokens.py` generates properly signed expiring JWTs for exactly the three assignment users. Copy the relevant token into the provided browser client. `/health` is a basic process check, not proof that an external model is up or data is indexed.

## 3. Test with the provided HTML client

Double-click **`test_client.html`** (kept unchanged from the employer package). Set WebSocket URL to `ws://localhost:8000/ws/chat`, paste one JWT, click **Connect**, and ask questions. The client communicates with precisely the requested protocol.

First client frame:

```json
{"type":"auth","token":"<JWT from mint_tokens.py>"}
```

Expected authentication response:

```json
{"type":"auth_success","user_id":"emp-001","department":"hr","level":1}
```

Then:

```json
{"type":"message","text":"How many days of annual leave do full-time employees receive?"}
```

Illustrative model token frames (exact token boundaries vary by provider):

```json
{"type":"stream","text":"Full-time employees "}
{"type":"stream","text":"receive 22 working days of paid annual leave "}
{"type":"stream","text":"[leave_policy.pdf p.1]."}
{"type":"done"}
```

These are **protocol illustrations, not captured claims of a successful live provider invocation**. An invalid/expired/missing JWT instead receives `auth_failed` and a policy-close (1008). A provider or tool error produces a readable message and `done`, while leaving the socket available for another request.

### Useful assessment walkthrough

| JWT | Ask | What to verify |
|---|---|---|
| `emp-001` (HR L1) | “How many paid annual leave days?” | HR leave policy: **22 working days**, page 1 |
| `emp-001` (HR L1) | “Tell me the private performance rating distribution.” | No disclosure of manager-only policy |
| `mgr-002` (HR L2) | “What are the five performance rating categories?” | Manager policy can be retrieved and cited |
| `exec-003` (Exec L3) | “What is the APAC investment in the scanned strategy?” | Strategic plan through OCR: **approximately USD 45 million** |
| `exec-003` (Exec L3) | “What is the finance expense preapproval threshold?” | Should **not** disclose finance-only policy even though level is 3 |
| Any | “Who is my manager?” | Tool returns the **mock** manager John Smith using 3 concurrent requests |

The first three user definitions are fixed by the assignment. A finance-L1 role is also explicitly tested in the suite to confirm the shared-HR exception.

### Offline test without Ollama, Groq, or a Qdrant download

For **protocol and access-control demonstration only**, change the following in `.env`:

```dotenv
LLM_PROVIDER=mock
EMBEDDING_PROVIDER=hashing
VECTOR_STORE=sqlite
```

Then run `python -m app.ingestion`, `python mint_tokens.py`, and `uvicorn app.main:app --reload` as above. This mode **does not generate policy answers**: it returns an unmistakable `[DEMO ONLY]` message or mock employee context. Hashing vectors are lexical and SQLite similarity is slower; this mode must not be presented as a production LLM / semantic embedding demo.

## 4. Chunking strategy and justification

The chunking unit is one PDF **page**. Every chunk retains `department` (from the path), integer `access_level` (from a strict explicit file-to-level map), `source_file` (original basename), and one-based `page` for citations. Normalized page text is split into **900-character windows with ~150 characters of overlap**. The splitter prefers paragraph, sentence, line, then word boundaries late in the window; it never crosses pages.

This deliberately favors clear policy clauses and citations over arbitrarily large snippets. A roughly 900-character slice fits several connected policy sentences; the overlap reduces loss when an entitlement or condition spans two slices. Page-specific metadata avoids incorrect page citations when searching PDFs with more than one page. The characters-based choice is model-agnostic; an improvement for longer complex policies would be section-, table-, and tokenizer-aware segmentation.

## 5. Retrieval-time RBAC (security-critical)

The JWT is verified with HS256 and validated for required identity/role/time claims before chat is permitted. The client cannot set or override the department or level. The **vector database itself** receives this logical filter:

```text
access_level <= authenticated_user.level
AND
(department == authenticated_user.department OR department == "hr")
```

In the Qdrant backend this is a `models.Filter` with a mandatory range condition and a nested `should` (OR) department filter, **passed directly in `query_points(query_filter=...)`**. Restricted chunks cannot enter the retrieval candidates, similarity ranking, top-K results, or LLM context. No Python list is fetched and filtered after similarity search. The optional SQLite test backend applies the identical predicate in its SQL `WHERE` clause, before similarity sorting and limiting.

| Identity | HR level 1 | HR level 2 | Finance level 1 | Exec level 3 |
|---|:---:|:---:|:---:|:---:|
| `emp-001`, HR L1 | Yes | No | No | No |
| `mgr-002`, HR L2 | Yes | Yes | No | No |
| `exec-003`, Exec L3 | Yes | Yes | No | Yes |
| Finance L1 (additional test) | Yes | No | Yes | No |

New PDF filenames without a defined access level cause ingestion to **fail closed**; they never silently default to public. Missing mandatory originals also prevent incomplete indexing. The tool's target `user_id` must exactly match the authenticated JWT subject; even a malicious model-generated function call cannot retrieve another employee's details.

## 6. Async/parallel approach and real streaming

The LLM receives **one function**: `get_employee_context(user_id)`. The LLM can choose it with a JSON tool call when the user's question concerns a grade, manager, or team. Only the identity from the validated JWT can be used. The tool runs three mock services concurrently:

```python
profile, manager, team = await asyncio.gather(
    _profile(user_id), _manager(user_id), _team(user_id)
)
```

Each coroutine awaits `asyncio.sleep(1)`; they run concurrently, making aggregate latency approximately **one second**, rather than three seconds from sequential requests. Its mock output is Jane Doe / Senior, John Smith, and Platform / team size 8. It does **not** contain a personal leave balance, so the prompt forbids inventing one.

The LLM first makes a short **non-streaming** tool-selection call; if selected, it collects the tool result, then requests the final answer using `stream=true`. This second call's HTTP Server-Sent Event `delta.content` chunks are forwarded immediately as WebSocket `stream` frames. It is **not** a fake word-by-word playback of an already completed LLM answer. `httpx.AsyncClient` handles network I/O without blocking. CPU-heavy embeddings and synchronous local vector DB operations are offloaded with `asyncio.to_thread`; PDF extraction runs in the ingestion CLI and off the WebSocket event loop.

## 7. One improvement given more time

Build a **measured answer-grounding and retrieval evaluation set** from the employer documents: questions with gold source pages (including OCR and negative access cases), then compare top-K recall, answer citation fidelity, and refusal quality with section-aware chunking and optionally hybrid lexical+dense retrieval or a lightweight reranker. Automated tests here establish *mechanics and controls*, not an unmeasured guarantee of LLM accuracy.

## 8. One deliberate tradeoff and why

**Embedded local Qdrant plus compact CPU-friendly embeddings** were chosen for free, reproducible review without Docker or SaaS accounts. The tradeoff is local DB locking/multi-process limitations, initial model download, and potentially lower recall than larger/hybrid systems. A second conscious tradeoff is the two-pass LLM flow for cross-provider tool compatibility, at the cost of an extra model round trip. For a deployed company system, host Qdrant as a managed or networked service, use `wss://`, add audit logging and explicit user/tenant authorization, secret rotation, and request rate limits.

## Verification evidence and limitations

Run the automated suite with a minimal lightweight environment:

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

The suite covers signed and expired JWTs, first-frame authentication enforcement, WebSocket message/stream/done behavior, repeated requests and redacted LLM/tool errors, fake provider SSE deltas, malicious cross-identity tool requests, parallel elapsed time, OCR on both generated fixtures **and the original scanned PDF**, ingestion metadata and source facts, Qdrant query filter **construction** (using a lightweight contract double), plus SQLite SQL-layer filtering of all original documents. A [GitHub Actions workflow](.github/workflows/tests.yml) runs the same tests with Tesseract installed. Three earlier development commits plus the final integration commits are included in the ZIP's Git history.

**Verified in this environment:** all **22 automated tests passed**; OCR recovered text from the original scanned strategy; the original seven PDFs produced **18 indexed chunks** in the offline SQL index; and a separately launched Uvicorn WebSocket server successfully handled original-client protocol frames for valid and invalid JWTs, continued between questions, streamed multiple frames, returned parallel mock-tool data, and rejected an invalid token with close code 1008.

**Not verified here:** an actual live answer from Ollama/Groq, a downloaded sentence-transformer inference run, or execution against the installed Qdrant package, because external package downloads and a live model service were unavailable. The Qdrant filter call is also tested via a fake client. The real-provider setup is implemented and documented, but run at least one actual provider+Qdrant smoke test on a machine with dependencies installed before representing generative-answer quality as validated. The system prompt requires retrieved, authorized evidence and citations; a generative model can still be mistaken, so accurate policy answers are not guaranteed.

The demo PDF generator under `scripts/` exists **only for isolated tests**; it is never invoked in the normal ingestion path and deliberately refuses to overwrite any of the employer-supplied documents. In contrast, `docs/` contains the original employer PDFs as provided. The browser test client is the unmodified employer file.

## Source layout

```text
.
├── app/
│   ├── main.py                # WebSocket protocol, JWT handshake, errors
│   ├── security.py            # Signature/expiry/role validation
│   ├── chat.py                # ACL-aware retrieval, prompt, tool orchestration
│   ├── llm.py                 # Real async SSE adapters for Ollama/Groq; marked mock
│   ├── tools.py               # Three concurrent mock services; single function schema
│   ├── embeddings.py          # Sentence-transformers + deterministic test vectors
│   ├── store.py               # Qdrant query-filtered vectors + SQL test alternative
│   ├── ingestion.py           # Original PDF text/OCR, chunks and CLI
│   └── config.py              # Environment settings
├── docs/{hr,finance,exec}/    # All SEVEN original policy PDFs
├── tests/                     # Contract, integration, OCR, security and async tests
├── scripts/create_demo_docs.py# Separate synthetic fixtures for isolated tests only
├── .github/workflows/tests.yml# Automated CI for offline checks
├── .env.example               # Non-secret configuration example
├── requirements.txt           # Runtime with Qdrant and semantic embeddings
├── requirements-test.txt      # Lightweight test-only environment
├── mint_tokens.py             # Signed test-user JWTs
├── test_client.html           # Employer's original browser test client
├── ASSIGNMENT_README.md       # Employer's original problem statement
└── README_SUBMISSION.md       # This document
```
