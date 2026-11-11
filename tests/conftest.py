import os

# Settings are read from the environment; tests never touch a real database or API.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("SERVICE_TOKEN", "test-token")
os.environ.setdefault("SPRING_BASE_URL", "http://spring.test")
for key in ("GEMINI_API_KEY", "LLM_API_KEY", "VOYAGE_API_KEY"):
    os.environ.pop(key, None)
