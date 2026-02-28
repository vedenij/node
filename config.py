"""
Configuration for the persistent worker node.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Settings loaded from environment variables."""

    # vLLM connection (co-located on same server)
    vllm_host: str = Field(default="127.0.0.1", description="vLLM server host")
    vllm_port: int = Field(default=8000, description="vLLM server port")

    # Fixed model params
    model_name: str = Field(
        default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8",
        description="Model name for vLLM",
    )
    seq_len: int = Field(default=1024, description="Sequence length")
    k_dim: int = Field(default=12, description="Output vector dimensions")

    # Security
    api_key: str = Field(default="", description="API key for auth")

    # Server
    host: str = Field(default="0.0.0.0", description="Listen host")
    port: int = Field(default=9000, description="Listen port")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
