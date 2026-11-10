#!/bin/sh
# Container entrypoint. Render's free plan has no pre-deploy command, so migrations run
# here; they're idempotent, so a restart is harmless.
set -e

alembic upgrade head
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --proxy-headers --forwarded-allow-ips='*'
