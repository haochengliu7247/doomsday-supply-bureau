from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "末日物资鉴定局"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=7861, ge=1, le=65535)
    app_share: bool = False
    log_level: str = "INFO"

    mock_mode: bool = True
    vlm_provider: Literal["mock", "ollama", "modelscope", "openai_compatible"] = "mock"
    image_provider: Literal["mock", "comfyui", "remote"] = "mock"

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3-vl:8b-instruct-q8_0"
    ollama_timeout_seconds: float = Field(default=120, gt=0)
    ollama_num_ctx: int = Field(default=8192, ge=2048, le=32768)
    ollama_keep_alive: str = "0"

    comfyui_base_url: str = "http://127.0.0.1:8188"
    comfyui_workflow_path: Path = PROJECT_ROOT / "workflows/flux2_klein_4b_api.json"
    comfyui_text_workflow_path: Path = (
        PROJECT_ROOT / "workflows/flux2_klein_4b_text_to_image_api.json"
    )
    comfyui_timeout_seconds: float = Field(default=180, gt=0)
    comfyui_poll_interval_seconds: float = Field(default=0.75, ge=0.1, le=5)
    comfyui_max_output_mb: int = Field(default=30, ge=1, le=100)
    comfyui_min_free_vram_mb: int = Field(default=20000, ge=1024, le=64000)
    image_identity_max_attempts: int = Field(default=2, ge=1, le=3)
    image_identity_min_confidence: float = Field(default=0.8, ge=0.5, le=1)

    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'doomsday.db').as_posix()}"
    output_dir: Path = PROJECT_ROOT / "outputs"
    scan_cache_enabled: bool = True
    ai_pipeline_lock_path: Path = PROJECT_ROOT / "data" / "ai_pipeline.lock"
    ai_pipeline_lock_timeout_seconds: float = Field(default=600, gt=0, le=3600)
    max_upload_mb: int = Field(default=15, ge=1, le=100)
    image_max_side: int = Field(default=1536, ge=512, le=4096)

    @field_validator("ollama_base_url", "comfyui_base_url")
    @classmethod
    def trim_url(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator(
        "output_dir",
        "ai_pipeline_lock_path",
        "comfyui_workflow_path",
        "comfyui_text_workflow_path",
    )
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        return value if value.is_absolute() else PROJECT_ROOT / value

    @field_validator("database_url")
    @classmethod
    def resolve_sqlite_url(cls, value: str) -> str:
        prefix = "sqlite:///"
        if not value.startswith(prefix):
            return value
        path = Path(value[len(prefix) :])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return f"{prefix}{path.resolve().as_posix()}"

    def ensure_directories(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (PROJECT_ROOT / "data").mkdir(parents=True, exist_ok=True)
        self.ai_pipeline_lock_path.parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
