from functools import lru_cache
from typing import List
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    ENV: str = "development"
    APP_NAME: str = "OPS AI GRID"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = True

    SECRET_KEY: str = Field(min_length=32)
    JWT_ACCESS_TTL_MINUTES: int = 15
    JWT_REFRESH_TTL_DAYS: int = 7
    JWT_ALGORITHM: str = "HS256"

    FERNET_KEY: str

    DATABASE_URL: str = "sqlite+aiosqlite:///./ops_ai_grid.db"

    QUEUE_BACKEND: str = "inprocess"
    REDIS_URL: str = ""

    RATE_LIMIT_PER_MINUTE: int = 60
    LOGIN_RATE_LIMIT_PER_MINUTE: int = 5

    ANTHROPIC_API_KEY: str = ""

    CORS_ORIGINS_RAW: str = Field(default="http://localhost:3000", alias="CORS_ORIGINS")

    @computed_field
    @property
    def CORS_ORIGINS(self) -> List[str]:
        return [o.strip() for o in self.CORS_ORIGINS_RAW.split(",") if o.strip()]

    STORAGE_BACKEND: str = "local"
    STORAGE_LOCAL_PATH: str = "./storage"
    S3_ENDPOINT: str = ""
    S3_BUCKET: str = ""
    S3_ACCESS_KEY: str = ""
    S3_SECRET_KEY: str = ""

    ADMIN_EMAIL: str = "admin@opsaigrid.local"
    ADMIN_PASSWORD: str = "ChangeMe!2026"
    ADMIN_TENANT_NAME: str = "OPS AI GRID HQ"

    @property
    def is_sqlite(self) -> bool:
        return self.DATABASE_URL.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return "postgresql" in self.DATABASE_URL


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
