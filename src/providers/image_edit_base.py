from pathlib import Path
from typing import Protocol

from src.schemas import AppraisalResult, ScanRequest


class ImageEditProvider(Protocol):
    @property
    def name(self) -> str: ...

    def edit(
        self,
        image_path: Path | None,
        appraisal: AppraisalResult,
        request: ScanRequest,
    ) -> Path | None: ...

    def unload(self) -> None: ...

