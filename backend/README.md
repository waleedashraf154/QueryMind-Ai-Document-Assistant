# QueryMind Backend

FastAPI backend for QueryMind.

## Run

From the project root:

```bash
python -m uvicorn backend.main:app --reload --port 8000
```

## Important after replacing this backend

1. Keep your existing `backend/.env` file. This archive intentionally does not include `.env`.
2. Restart the backend after replacing the files.
3. Re-upload/re-index existing documents after the update so old chunk/embedding data is replaced with the corrected extraction and section metadata.

## RAG behavior

- Numbered document headings are extracted deterministically and kept exactly as written.
- Named-section requests retrieve the complete requested section instead of a fixed top-k slice.
- Follow-up requests such as "tell me more" and "remaining" inherit the previous document/section context and trigger fresh retrieval.
- Normal questions use broader hybrid retrieval settings and larger grounded context.
- Deep/relational questions are kept in normal RAG instead of being misclassified as simple topic lookups.
- `Section: PROFILE` / `Section: REFERENCE` style parser wrappers are not treated as real headings.
- `Reference Table A/B/C/D` is preserved as section content, not promoted to a document heading.

## Structure

- `routes/` — FastAPI HTTP endpoints
- `schemas/` — request/response models
- `services/` — application/business logic
- `database/` — persistence access
- `parsers/` — document and image extraction
- `rag_pipeline/` — chunking, retrieval, query routing, generation, and vector storage
- `models/` — runtime face-model assets supplied by the local project
- `main.py` — FastAPI application setup and router registration
