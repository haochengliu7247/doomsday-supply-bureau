from dataclasses import dataclass
from enum import StrEnum


class ErrorCode(StrEnum):
    INVALID_INPUT = "invalid_input"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TIMEOUT = "timeout"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    GPU_OOM = "gpu_oom"
    IMAGE_EDIT_FAILED = "image_edit_failed"
    INTERNAL = "internal"


class PipelineStage(StrEnum):
    INPUT = "INPUT"
    VLM = "VLM"
    IMAGE_EDIT = "IMAGE_EDIT"
    GAME = "GAME"
    DATABASE = "DATABASE"


@dataclass(slots=True)
class AppError(Exception):
    code: ErrorCode
    stage: PipelineStage
    user_message: str
    retriable: bool = False
    detail: str = ""

    def __str__(self) -> str:
        return self.user_message

