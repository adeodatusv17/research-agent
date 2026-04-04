from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5432/research_agent"
    async_database_url: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/research_agent"
    )
    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-large"
    artifacts_dir: str = "./artifacts"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
