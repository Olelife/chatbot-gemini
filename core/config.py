from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_ID: str = "olelifetech"
    LOCATION: str = "us-central1"

    BUCKET: str = "olelife-lakehouse"
    KNOWLEDGE_FOLDER: str = "gemini-ai/knowledge"

    EMBED_MODEL: str = "gemini-embedding-001"
    GEN_MODEL: str = "gemini-2.5-flash"

    CACHE_LOCAL_PATH: str = "/tmp/embedding_cache.pkl"
    CACHE_GCS_PATH: str = "gemini-ai/embedding_cache.pkl"

    API_CHAT_GEMINI_DB_HOST: str
    API_CHAT_GEMINI_DB_USER: str
    API_CHAT_GEMINI_DB_PASS: str
    API_CHAT_GEMINI_DB_NAME: str = "ole-db-ia"

    class Config:
        env_file = ".env"


settings = Settings()