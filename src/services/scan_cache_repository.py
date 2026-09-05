from __future__ import annotations

import hashlib
import json
import sqlite3
import unicodedata
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from PIL import Image, UnidentifiedImageError

from src.config import Settings
from src.errors import AppError, ErrorCode, PipelineStage
from src.providers.comfyui_provider import (
    IMAGE_EDIT_PROMPT_VERSION,
    TEXT_TO_IMAGE_PROMPT_VERSION,
)
from src.providers.ollama_provider import (
    APPRAISAL_PROMPT_VERSION,
    IDENTITY_PROMPT_VERSION,
)
from src.schemas import (
    AppraisalPayload,
    AppraisalResult,
    ImageIdentityVerdict,
    InventoryItem,
    PipelineResult,
    PipelineStatus,
    ScanRequest,
)
from src.services.image_service import IMAGE_NORMALIZATION_VERSION, MAX_INPUT_PIXELS

CACHE_CONTRACT_VERSION = "scan-cache-v1"
CACHE_ACCEPTANCE_VERSION = "complete-success-v1"
_ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    cache_key: str
    input_mode: Literal["text", "photo"]
    normalized_description: str
    input_image_sha256: str | None
    scenario: str
    apocalypse_years_key: str
    pipeline_signature: str


@dataclass(frozen=True, slots=True)
class PrewarmItem:
    lexicon_version: str
    ordinal: int
    category: str
    canonical_description: str
    attempts: int


def normalize_description(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value or "").split()).casefold()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ScanCacheRepository:
    """SQLite metadata and validated local files for exact complete scan results."""

    def __init__(self, settings: Settings) -> None:
        prefix = "sqlite:///"
        if not settings.database_url.startswith(prefix):
            raise ValueError("完整结果缓存当前只支持 sqlite:/// 本地数据库。")
        raw_path = settings.database_url[len(prefix) :]
        if not raw_path:
            raise ValueError("缓存数据库路径不能为空。")
        self.settings = settings
        self.database_path = Path(raw_path).expanduser().resolve()
        self.output_root = settings.output_dir.resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        try:
            with self._connect() as connection:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS scan_cache (
                        cache_key TEXT PRIMARY KEY CHECK(length(cache_key) = 64),
                        input_mode TEXT NOT NULL
                            CHECK(input_mode IN ('text', 'photo')),
                        normalized_description TEXT NOT NULL,
                        input_image_sha256 TEXT,
                        scenario TEXT NOT NULL,
                        apocalypse_years_key TEXT NOT NULL,
                        pipeline_signature TEXT NOT NULL,
                        appraisal_json TEXT NOT NULL,
                        before_relpath TEXT,
                        before_sha256 TEXT,
                        after_relpath TEXT NOT NULL,
                        after_sha256 TEXT NOT NULL,
                        provider_metadata_json TEXT NOT NULL DEFAULT '{}',
                        source TEXT NOT NULL DEFAULT 'runtime'
                            CHECK(source IN ('runtime', 'prewarm')),
                        created_at TEXT NOT NULL,
                        last_accessed_at TEXT NOT NULL,
                        hit_count INTEGER NOT NULL DEFAULT 0
                            CHECK(hit_count >= 0),
                        CHECK(
                            (input_mode = 'text'
                                AND input_image_sha256 IS NULL
                                AND before_relpath IS NULL
                                AND before_sha256 IS NULL)
                            OR
                            (input_mode = 'photo'
                                AND input_image_sha256 IS NOT NULL
                                AND before_relpath IS NOT NULL
                                AND before_sha256 IS NOT NULL)
                        )
                    );

                    CREATE INDEX IF NOT EXISTS idx_scan_cache_last_accessed
                    ON scan_cache(last_accessed_at);

                    CREATE INDEX IF NOT EXISTS idx_scan_cache_signature_source
                    ON scan_cache(pipeline_signature, source);

                    CREATE TABLE IF NOT EXISTS scan_cache_aliases (
                        normalized_alias TEXT PRIMARY KEY,
                        canonical_description TEXT NOT NULL,
                        lexicon_version TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS scan_cache_prewarm (
                        lexicon_version TEXT NOT NULL,
                        ordinal INTEGER NOT NULL CHECK(ordinal >= 1),
                        category TEXT NOT NULL,
                        canonical_description TEXT NOT NULL,
                        pipeline_signature TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending'
                            CHECK(status IN ('pending', 'running', 'ready', 'failed')),
                        attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts >= 0),
                        cache_key TEXT,
                        duration_ms INTEGER,
                        last_error TEXT,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(lexicon_version, canonical_description),
                        UNIQUE(lexicon_version, ordinal)
                    );

                    CREATE INDEX IF NOT EXISTS idx_scan_cache_prewarm_status
                    ON scan_cache_prewarm(lexicon_version, status, ordinal);
                    """
                )
                connection.execute("PRAGMA optimize")
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def make_identity(
        self,
        request: ScanRequest,
        *,
        input_image_sha256: str | None,
    ) -> CacheIdentity:
        input_mode: Literal["text", "photo"] = (
            "photo" if input_image_sha256 is not None else "text"
        )
        normalized = normalize_description(request.description)
        if input_mode == "text":
            normalized = self.resolve_alias(normalized)
        years_key = format(request.apocalypse_years, ".12g")
        signature = self._pipeline_signature(input_mode)
        payload = {
            "cache_contract": CACHE_CONTRACT_VERSION,
            "input_mode": input_mode,
            "description": normalized,
            "image_sha256": input_image_sha256,
            "scenario": request.scenario.value,
            "years": years_key,
            "pipeline_signature": signature,
        }
        return CacheIdentity(
            cache_key=_sha256_bytes(_stable_json(payload).encode("utf-8")),
            input_mode=input_mode,
            normalized_description=normalized,
            input_image_sha256=input_image_sha256,
            scenario=request.scenario.value,
            apocalypse_years_key=years_key,
            pipeline_signature=signature,
        )

    def resolve_alias(self, normalized_description: str) -> str:
        if not normalized_description:
            return normalized_description
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT canonical_description
                    FROM scan_cache_aliases
                    WHERE normalized_alias = ?
                    """,
                    (normalized_description,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        return str(row["canonical_description"]) if row else normalized_description

    def lookup(
        self,
        identity: CacheIdentity,
        request: ScanRequest,
    ) -> PipelineResult | None:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT * FROM scan_cache WHERE cache_key = ?",
                    (identity.cache_key,),
                ).fetchone()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        if row is None:
            return None

        try:
            after_path = self._validated_cached_image(
                str(row["after_relpath"]), str(row["after_sha256"])
            )
            before_path = None
            if identity.input_mode == "photo":
                before_path = self._validated_cached_image(
                    str(row["before_relpath"]), str(row["before_sha256"])
                )
            appraisal = AppraisalResult.model_validate_json(str(row["appraisal_json"]))
            metadata = json.loads(str(row["provider_metadata_json"]))
            if not isinstance(metadata, dict):
                raise ValueError("provider metadata must be an object")
        except (OSError, ValueError, UnidentifiedImageError):
            self.invalidate(identity.cache_key)
            return None

        item = InventoryItem(
            appraisal=appraisal.model_copy(deep=True),
            original_image=str(before_path) if before_path else None,
            apocalypse_image=str(after_path),
            source_description=request.description.strip(),
            scenario=request.scenario,
            apocalypse_years=request.apocalypse_years,
        )
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE scan_cache
                    SET hit_count = hit_count + 1, last_accessed_at = ?
                    WHERE cache_key = ?
                    """,
                    (now, identity.cache_key),
                )
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        metadata = {
            **metadata,
            "cache_hit": True,
            "cache_key": identity.cache_key,
            "cache_source": str(row["source"]),
        }
        return PipelineResult(
            status=PipelineStatus.SUCCESS,
            item=item,
            original_image=item.original_image,
            apocalypse_image=item.apocalypse_image,
            timings_ms={"vlm": 0, "image_edit": 0, "total": 0},
            provider_metadata=metadata,
        )

    def store(
        self,
        identity: CacheIdentity,
        result: PipelineResult,
        *,
        source: Literal["runtime", "prewarm"] = "runtime",
    ) -> bool:
        if not self._is_cacheable(identity, result):
            return False
        assert result.apocalypse_image is not None
        after_path = Path(result.apocalypse_image)
        before_path = Path(result.original_image) if result.original_image else None
        after_relpath, after_sha256 = self._cache_file_record(after_path)
        before_relpath = None
        before_sha256 = None
        if before_path is not None:
            before_relpath, before_sha256 = self._cache_file_record(before_path)

        now = datetime.now(UTC).isoformat()
        metadata = {**result.provider_metadata, "cache_hit": False}
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO scan_cache(
                        cache_key, input_mode, normalized_description,
                        input_image_sha256, scenario, apocalypse_years_key,
                        pipeline_signature, appraisal_json, before_relpath,
                        before_sha256, after_relpath, after_sha256,
                        provider_metadata_json, source, created_at,
                        last_accessed_at, hit_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        appraisal_json = excluded.appraisal_json,
                        before_relpath = excluded.before_relpath,
                        before_sha256 = excluded.before_sha256,
                        after_relpath = excluded.after_relpath,
                        after_sha256 = excluded.after_sha256,
                        provider_metadata_json = excluded.provider_metadata_json,
                        source = excluded.source,
                        last_accessed_at = excluded.last_accessed_at
                    """,
                    (
                        identity.cache_key,
                        identity.input_mode,
                        identity.normalized_description,
                        identity.input_image_sha256,
                        identity.scenario,
                        identity.apocalypse_years_key,
                        identity.pipeline_signature,
                        result.item.appraisal.model_dump_json(),
                        before_relpath,
                        before_sha256,
                        after_relpath,
                        after_sha256,
                        _stable_json(metadata),
                        source,
                        now,
                        now,
                    ),
                )
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        return True

    def invalidate(self, cache_key: str) -> None:
        try:
            with self._connect() as connection:
                connection.execute("DELETE FROM scan_cache WHERE cache_key = ?", (cache_key,))
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def register_prewarm_manifest(
        self,
        *,
        lexicon_version: str,
        items: list[tuple[int, str, str]],
        aliases: dict[str, str],
        pipeline_signature: str,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.executemany(
                    """
                    INSERT INTO scan_cache_aliases(
                        normalized_alias, canonical_description, lexicon_version
                    ) VALUES (?, ?, ?)
                    ON CONFLICT(normalized_alias) DO UPDATE SET
                        canonical_description = excluded.canonical_description,
                        lexicon_version = excluded.lexicon_version
                    """,
                    [
                        (
                            normalize_description(alias),
                            normalize_description(canonical),
                            lexicon_version,
                        )
                        for alias, canonical in aliases.items()
                    ],
                )
                connection.executemany(
                    """
                    INSERT INTO scan_cache_prewarm(
                        lexicon_version, ordinal, category,
                        canonical_description, pipeline_signature, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(lexicon_version, canonical_description) DO UPDATE SET
                        category = excluded.category,
                        pipeline_signature = excluded.pipeline_signature,
                        status = CASE
                            WHEN scan_cache_prewarm.pipeline_signature
                                 = excluded.pipeline_signature
                            THEN scan_cache_prewarm.status
                            ELSE 'pending'
                        END,
                        attempts = CASE
                            WHEN scan_cache_prewarm.pipeline_signature
                                 = excluded.pipeline_signature
                            THEN scan_cache_prewarm.attempts
                            ELSE 0
                        END,
                        cache_key = CASE
                            WHEN scan_cache_prewarm.pipeline_signature
                                 = excluded.pipeline_signature
                            THEN scan_cache_prewarm.cache_key
                            ELSE NULL
                        END,
                        last_error = CASE
                            WHEN scan_cache_prewarm.pipeline_signature
                                 = excluded.pipeline_signature
                            THEN scan_cache_prewarm.last_error
                            ELSE 'pipeline signature changed; regeneration required'
                        END,
                        updated_at = excluded.updated_at
                    """,
                    [
                        (
                            lexicon_version,
                            ordinal,
                            category,
                            normalize_description(description),
                            pipeline_signature,
                            now,
                        )
                        for ordinal, category, description in items
                    ],
                )
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def recover_interrupted_prewarm(self, lexicon_version: str) -> int:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                cursor = connection.execute(
                    """
                    UPDATE scan_cache_prewarm
                    SET status = 'pending', last_error = 'worker interrupted; resumed',
                        updated_at = ?
                    WHERE lexicon_version = ? AND status = 'running'
                    """,
                    (now, lexicon_version),
                )
                connection.commit()
                return cursor.rowcount
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def claim_next_prewarm(
        self,
        lexicon_version: str,
        *,
        max_attempts: int,
    ) -> PrewarmItem | None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """
                    SELECT lexicon_version, ordinal, category,
                           canonical_description, attempts
                    FROM scan_cache_prewarm
                    WHERE lexicon_version = ?
                      AND (
                        status = 'pending'
                        OR (status = 'failed' AND attempts < ?)
                      )
                    ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END,
                             ordinal ASC
                    LIMIT 1
                    """,
                    (lexicon_version, max_attempts),
                ).fetchone()
                if row is None:
                    connection.commit()
                    return None
                attempts = int(row["attempts"]) + 1
                connection.execute(
                    """
                    UPDATE scan_cache_prewarm
                    SET status = 'running', attempts = ?, last_error = NULL,
                        updated_at = ?
                    WHERE lexicon_version = ? AND canonical_description = ?
                    """,
                    (
                        attempts,
                        now,
                        lexicon_version,
                        str(row["canonical_description"]),
                    ),
                )
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        return PrewarmItem(
            lexicon_version=str(row["lexicon_version"]),
            ordinal=int(row["ordinal"]),
            category=str(row["category"]),
            canonical_description=str(row["canonical_description"]),
            attempts=attempts,
        )

    def finish_prewarm(
        self,
        item: PrewarmItem,
        *,
        succeeded: bool,
        cache_key: str | None,
        duration_ms: int,
        error: str | None = None,
    ) -> None:
        now = datetime.now(UTC).isoformat()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    UPDATE scan_cache_prewarm
                    SET status = ?, cache_key = ?, duration_ms = ?,
                        last_error = ?, updated_at = ?
                    WHERE lexicon_version = ? AND canonical_description = ?
                    """,
                    (
                        "ready" if succeeded else "failed",
                        cache_key,
                        duration_ms,
                        None if succeeded else (error or "unknown error")[:2000],
                        now,
                        item.lexicon_version,
                        item.canonical_description,
                    ),
                )
                connection.commit()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def prewarm_progress(self, lexicon_version: str) -> dict[str, int]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT status, COUNT(*) AS count
                    FROM scan_cache_prewarm
                    WHERE lexicon_version = ?
                    GROUP BY status
                    """,
                    (lexicon_version,),
                ).fetchall()
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc
        counts = {"pending": 0, "running": 0, "ready": 0, "failed": 0}
        counts.update({str(row["status"]): int(row["count"]) for row in rows})
        counts["total"] = sum(counts.values())
        return counts

    def cache_count(self) -> int:
        try:
            with self._connect() as connection:
                return int(connection.execute("SELECT COUNT(*) FROM scan_cache").fetchone()[0])
        except sqlite3.DatabaseError as exc:
            raise self._database_error(exc) from exc

    def _pipeline_signature(self, input_mode: Literal["text", "photo"]) -> str:
        workflow_path = (
            self.settings.comfyui_text_workflow_path
            if input_mode == "text"
            else self.settings.comfyui_workflow_path
        )
        workflow_hash = (
            _sha256_file(workflow_path)
            if workflow_path.is_file()
            else f"missing:{workflow_path.name}"
        )
        payload: dict[str, Any] = {
            "cache_contract": CACHE_CONTRACT_VERSION,
            "acceptance": CACHE_ACCEPTANCE_VERSION,
            "input_mode": input_mode,
            "vlm_provider": self.settings.vlm_provider,
            "vlm_model": self.settings.ollama_model,
            "image_provider": self.settings.image_provider,
            "workflow_sha256": workflow_hash,
            "appraisal_prompt_version": APPRAISAL_PROMPT_VERSION,
            "appraisal_schema_sha256": _sha256_bytes(
                _stable_json(AppraisalPayload.model_json_schema()).encode("utf-8")
            ),
            "text_to_image_prompt_version": TEXT_TO_IMAGE_PROMPT_VERSION,
        }
        if input_mode == "photo":
            payload.update(
                {
                    "identity_prompt_version": IDENTITY_PROMPT_VERSION,
                    "identity_schema_sha256": _sha256_bytes(
                        _stable_json(ImageIdentityVerdict.model_json_schema()).encode("utf-8")
                    ),
                    "image_edit_prompt_version": IMAGE_EDIT_PROMPT_VERSION,
                    "identity_min_confidence": self.settings.image_identity_min_confidence,
                    "image_normalization_version": IMAGE_NORMALIZATION_VERSION,
                    "image_max_side": self.settings.image_max_side,
                }
            )
        return _sha256_bytes(_stable_json(payload).encode("utf-8"))

    def _is_cacheable(self, identity: CacheIdentity, result: PipelineResult) -> bool:
        if result.status is not PipelineStatus.SUCCESS:
            return False
        if result.item.appraisal.is_fallback or not result.apocalypse_image:
            return False
        if identity.input_mode == "photo":
            return (
                bool(result.original_image)
                and result.provider_metadata.get("identity_verified") is True
            )
        return result.original_image is None

    def _cache_file_record(self, path: Path) -> tuple[str, str]:
        resolved = path.resolve(strict=True)
        resolved.relative_to(self.output_root)
        self._verify_image(resolved)
        relpath = resolved.relative_to(self.output_root).as_posix()
        return relpath, _sha256_file(resolved)

    def _validated_cached_image(self, relpath: str, expected_sha256: str) -> Path:
        relative = PurePosixPath(relpath)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("unsafe cache image path")
        candidate = (self.output_root / Path(*relative.parts)).resolve(strict=True)
        candidate.relative_to(self.output_root)
        self._verify_image(candidate)
        if _sha256_file(candidate) != expected_sha256:
            raise ValueError("cache image checksum mismatch")
        return candidate

    @staticmethod
    def _verify_image(path: Path) -> None:
        if path.suffix.lower() not in _ALLOWED_IMAGE_SUFFIXES or path.stat().st_size <= 0:
            raise ValueError("unsupported or empty cache image")
        with Image.open(path) as image:
            if image.width * image.height > MAX_INPUT_PIXELS:
                raise ValueError("cache image exceeds pixel limit")
            image.verify()

    @staticmethod
    def _database_error(exc: sqlite3.DatabaseError) -> AppError:
        return AppError(
            code=ErrorCode.INTERNAL,
            stage=PipelineStage.DATABASE,
            user_message="本地完整结果缓存暂时无法读写，本次将继续使用模型。",
            retriable=True,
            detail=str(exc),
        )
