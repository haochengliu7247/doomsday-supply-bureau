from pathlib import Path
from typing import Protocol

from src.schemas import AppraisalResult, ImageIdentityVerdict, ScanRequest


class VLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    def analyze(
        self,
        image_path: Path | None,
        request: ScanRequest,
    ) -> AppraisalResult: ...

    def verify_identity(
        self,
        source_path: Path,
        candidate_path: Path,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> ImageIdentityVerdict: ...

    def unload(self) -> None: ...
