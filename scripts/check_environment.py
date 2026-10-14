"""Read-only preflight checks for reviewers running the real (not mock) backend.

Run from the repository root: python -m scripts.check_environment
No packages or models are installed by this script.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

from app.config import get_settings
from app.ingestion import FILE_ACCESS_LEVELS


def main() -> int:
    settings = get_settings()
    errors: list[str] = []
    print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] not in {(3, 11), (3, 12)}:
        print("NOTE: Python 3.11 or 3.12 is the documented, recommended runtime.")
    if not Path('.env').is_file():
        errors.append("Missing .env; copy .env.example to .env and set a JWT_SECRET.")
    secret = settings.jwt_secret.get_secret_value()
    if not secret or secret == 'change-this-to-a-long-random-secret':
        errors.append('Set a unique, non-default JWT_SECRET in .env.')
    else:
        print('JWT secret: configured (not displayed)')

    missing = sorted(rel for rel in FILE_ACCESS_LEVELS if not (settings.docs_dir / rel).is_file())
    if missing:
        errors.append('Missing policy PDFs: ' + ', '.join(missing))
    else:
        print(f'PDFs: all {len(FILE_ACCESS_LEVELS)} required originals present')

    if not shutil.which('tesseract'):
        errors.append('Tesseract OCR is not on PATH; install it and reopen the VS Code terminal.')
    else:
        print('Tesseract OCR: found')

    for provider, module in ((settings.vector_store, 'qdrant_client' if settings.vector_store == 'qdrant' else 'sqlite3'),
                             (settings.embedding_provider, 'sentence_transformers' if settings.embedding_provider == 'sentence_transformers' else 'hashlib')):
        if importlib.util.find_spec(module) is None:
            errors.append(f'Missing package for {provider}: {module}; run pip install -r requirements.txt')
        else:
            print(f'{provider}: {module} module available')

    if settings.llm_provider == 'ollama':
        url = settings.ollama_base_url.removesuffix('/v1').rstrip('/') + '/api/tags'
        try:
            with urlopen(url, timeout=4) as response:
                tags = json.load(response).get('models', [])
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            errors.append(f'Ollama is not responding at {url}: {type(exc).__name__}; start Ollama.')
        else:
            names = {tag.get('name') for tag in tags} | {tag.get('model') for tag in tags}
            if settings.ollama_model not in names:
                errors.append(f'Ollama model {settings.ollama_model!r} is not installed; run: ollama pull {settings.ollama_model}')
                print('Installed Ollama tags:', ', '.join(sorted(name for name in names if name)) or '(none)')
            else:
                print(f'Ollama model: {settings.ollama_model} available locally')
    elif settings.llm_provider == 'groq':
        if not settings.groq_api_key.get_secret_value():
            errors.append('Groq selected but GROQ_API_KEY is missing.')
        else:
            print(f'Groq: key configured; model {settings.groq_model} (remote availability not checked)')
    else:
        print('WARNING: Mock LLM selected; not a production/generative demo.')

    print('\n' + ('READY for ingestion/server startup.' if not errors else 'ACTION REQUIRED:'))
    for error in errors:
        print('- ' + error)
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
