"""Build the embedding and LLM clients from settings (None when the key is missing)."""

from app.clients.embeddings import Embedder, GeminiEmbedder, VoyageEmbedder
from app.clients.llm import LLM
from app.config import Settings


def build_embedder(settings: Settings) -> Embedder | None:
    if settings.embedding_provider == "gemini":
        if settings.gemini_api_key is None:
            return None
        return GeminiEmbedder(
            settings.gemini_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_batch_size,
        )
    if settings.embedding_provider == "voyage":
        if settings.voyage_api_key is None:
            return None
        return VoyageEmbedder(
            settings.voyage_api_key.get_secret_value(),
            settings.embedding_model,
            settings.embedding_dim,
            settings.embedding_batch_size,
        )
    raise ValueError(f"Unknown EMBEDDING_PROVIDER {settings.embedding_provider!r} (use gemini or voyage)")


def missing_embedder_key(settings: Settings) -> str:
    return "VOYAGE_API_KEY" if settings.embedding_provider == "voyage" else "GEMINI_API_KEY"


def build_llm(settings: Settings) -> LLM | None:
    key = settings.effective_llm_api_key
    if key is None:
        return None
    return LLM(key.get_secret_value(), settings.llm_model, settings.llm_base_url)
