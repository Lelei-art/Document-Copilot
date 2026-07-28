from pathlib import Path

from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


_BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str

    # Database
    database_url: str

    # Ollama/local model settings.
    # These are only used when ASSISTANT_MODE=llm.
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    embedding_model: str = "nomic-embed-text:latest"
    embedding_dimensions: int = 768
    ollama_agent_request_limit: int = 8

    # Use "retrieval" on Railway if you do not want paid AI APIs.
    assistant_mode: str = "llm"

    # Retrieval settings
    retrieval_candidate_k: int = 50
    retrieval_top_k: int = 10
    retrieval_rrf_k: int = 60
    retrieval_neighbor_radius: int = 1
    retrieval_fts_config: str = "english"
    retrieval_fts_keyword_min: int = 3
    retrieval_fts_keyword_max: int = 5
    retrieval_fts_keyword_fast_path_tokens: int = 5

    # CORS
    allowed_origins: str = "http://localhost:5173"

    @computed_field
    @property
    def sqlalchemy_database_url(self) -> str:
        url = self.database_url

        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+psycopg://", 1)

        if url.startswith("postgres://"):
            return url.replace("postgres://", "postgresql+psycopg://", 1)

        return url

    @computed_field
    @property
    def cors_origins(self) -> list[str]:
        return [
            origin.strip()
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]


settings = Settings()
