from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: str | None = None
    COLLECTION_NAME: str = "documents"
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    GEMINI_MODEL: str = "gemini-1.5-flash"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    class Config:
        env_file = ".env"

settings = Settings()