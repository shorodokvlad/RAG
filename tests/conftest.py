import os

# Settings are read from the environment; tests never touch a real database or API.
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/test")
os.environ.setdefault("SERVICE_TOKEN", "test-token")
os.environ.setdefault("SPRING_BASE_URL", "http://spring.test")
os.environ.pop("VOYAGE_API_KEY", None)
