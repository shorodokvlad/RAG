FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Dependencies first so code changes don't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir app && touch app/__init__.py && pip install . && rm -rf app

COPY app ./app
COPY alembic.ini ./
COPY alembic ./alembic
COPY kb ./kb
COPY scripts/start.sh ./scripts/start.sh
RUN pip install --no-deps . \
    && useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin shv

USER shv
EXPOSE 8000
CMD ["sh", "scripts/start.sh"]
