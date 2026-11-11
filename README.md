# SHV Assistant: AI service

The RAG / LLM service behind the AI features of
[spring-ecommerce-app](https://github.com/shorodokvlad/spring-ecommerce-app): the chat
assistant, AI product overviews and review summaries. The full roadmap is in [PLAN.md](PLAN.md).

**Status: Phase 1 (product page AI).** Catalog indexing, AI product overviews, AI review
summaries and product Q&A with citations. Everything runs on **free tiers**: Gemini for
embeddings and text generation, Supabase pgvector for search, Render for hosting.

```
Spring (/internal/ai/**) ──► SpringClient ──► chunkers ──► Gemini embeddings ──► ai.chunks (pgvector, Supabase)
            ▲                                                                          
            └── X-Service-Token ◄── this service ◄── POST /index/products/{id} (from Spring, after commit)
```

## Requirements

- Python 3.12 or 3.13
- The Spring backend running with `AI_SERVICE_TOKEN` set to the same value as `SERVICE_TOKEN` here
- A Postgres database with pgvector: your Supabase project, or the local `docker compose` one
- A free [Google AI Studio](https://aistudio.google.com/apikey) API key (`GEMINI_API_KEY`) for
  embeddings and the LLM. Check your free quota at https://aistudio.google.com/rate-limit.
  On the free tier Google may use prompts to improve its products; everything sent is public
  product data, reviews and shoppers' questions, never personal data.

## Setup

```sh
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env        # fill in DATABASE_URL, SERVICE_TOKEN, SPRING_BASE_URL, GEMINI_API_KEY
alembic upgrade head        # creates the `ai` schema + tables, and enables pgvector
```

If you use [uv](https://docs.astral.sh/uv/), run `uv sync --extra dev` instead of the venv and pip steps.

## Index the catalog

```sh
python -m app.indexing.backfill --dry-run --product-id 1   # see the chunks; no DB or API key needed
python -m app.indexing.backfill                             # embed and store everything
```

Re-running it is cheap: only chunks whose text changed are re-embedded.
Check the result in Supabase's SQL editor:

```sql
-- nearest neighbours of product 1's card
SELECT c2.product_id, c2.metadata->>'product_name' AS name,
       round((1 - (c1.embedding <=> c2.embedding))::numeric, 3) AS similarity
FROM ai.chunks c1
JOIN ai.chunks c2 ON c2.source_type = 'product_card' AND c2.product_id <> c1.product_id
WHERE c1.source_type = 'product_card' AND c1.product_id = 1
ORDER BY c1.embedding <=> c2.embedding
LIMIT 5;
```

## Run the API

```sh
uvicorn app.main:app --reload --port 8000
```

| Endpoint | Auth | Purpose |
|---|---|---|
| `GET /health` | none | Liveness (used by Render) |
| `GET /health/ready` | token | Database check + chunk counts |
| `POST /index/products/{id}` | token | Re-index one product in the background (deletes its chunks if it no longer exists) |
| `DELETE /index/products/{id}` | token | Drop a product's chunks |
| `POST /index/reconcile` | token | Full re-index + prune (nightly GitHub Action) |
| `GET /products/{id}/overview` | token | AI overview (generated once, regenerated when the product text changes) |
| `GET /products/{id}/review-summary` | token | AI review summary (needs 3+ reviews; at most one regeneration per 10 minutes) |
| `POST /products/{id}/ask` | token | `{"question": "..."}` → answer with citations |

Browsers never call these directly: Spring's `/ai/products/**` gateway does, with caching and
rate limiting. If the free LLM quota runs out, these answer 503 and the storefront simply hides
the AI cards.

### Switching LLM provider

The LLM client speaks the OpenAI chat format, so any compatible free tier works. Set
`LLM_BASE_URL`, `LLM_API_KEY` and `LLM_MODEL` (see `.env.example`). The model must support
JSON-schema structured output. Embeddings stay on Gemini unless you set
`EMBEDDING_PROVIDER=voyage`, which re-embeds everything on the next reconcile.

"token" means the `X-Service-Token: $SERVICE_TOKEN` header.

## Tests

```sh
pytest -q && ruff check . && ruff format --check .
```

## Deploy on Render (free)

1. Render → **New + → Blueprint** → choose this repo. `render.yaml` creates the `shv-ai` web
   service in Frankfurt.
2. Fill in the prompted secrets:
   - `DATABASE_URL`: the Supabase **Session pooler** URI. Render has no IPv6, so use the pooler, not the direct connection.
   - `SERVICE_TOKEN`
   - `SPRING_BASE_URL`
   - `GEMINI_API_KEY`
3. On the Spring service, set `AI_SERVICE_URL` to `https://shv-ai.onrender.com` (your URL) and
   `AI_SERVICE_TOKEN` to the same value as `SERVICE_TOKEN`.
4. In this GitHub repo, add the secrets `AI_SERVICE_URL` and `AI_SERVICE_TOKEN` to enable the nightly reconcile.
5. Run the first backfill once, either from your machine (step "Index the catalog" with the
   production `.env`) or with `curl -X POST -H "X-Service-Token: …" https://shv-ai.onrender.com/index/reconcile`.

Free instances sleep after about 15 minutes idle and take up to a minute to wake. Migrations run on every start (`scripts/start.sh`).
