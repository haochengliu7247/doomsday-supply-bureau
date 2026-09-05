from __future__ import annotations

import sqlite3
import unicodedata
from collections.abc import Iterable
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from src.errors import AppError, ErrorCode, PipelineStage
from src.schemas import GameState, MarketItem


class GameRepository:
    """Small SQLite repository for named local saves and the market catalog."""

    def __init__(self, database_url: str) -> None:
        prefix = "sqlite:///"
        if not database_url.startswith(prefix):
            raise ValueError("当前只支持 sqlite:/// 本地数据库。")
        raw_path = database_url[len(prefix) :]
        if not raw_path:
            raise ValueError("数据库路径不能为空。")
        self.database_path = Path(raw_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self):
        connection = sqlite3.connect(self.database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 5000")
            yield connection
        except sqlite3.DatabaseError as exc:
            raise AppError(
                code=ErrorCode.INTERNAL,
                stage=PipelineStage.DATABASE,
                user_message="本地存档暂时无法读写，请稍后重试。",
                retriable=True,
                detail=str(exc),
            ) from exc
        finally:
            connection.close()

    def initialize(self, catalog: Iterable[MarketItem]) -> None:
        items = list(catalog)
        if not items:
            raise ValueError("市场目录不能为空。")
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    name TEXT PRIMARY KEY,
                    normalized_name TEXT NOT NULL UNIQUE,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS market_items (
                    item_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    category TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    water_price REAL NOT NULL CHECK (water_price >= 0),
                    health_gain INTEGER NOT NULL DEFAULT 0,
                    satiety_gain INTEGER NOT NULL DEFAULT 0,
                    energy_gain INTEGER NOT NULL DEFAULT 0,
                    morale_gain INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS market_items_category_idx
                ON market_items(category);

                CREATE TABLE IF NOT EXISTS app_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            placeholders = ",".join("?" for _ in items)
            connection.execute(
                f"DELETE FROM market_items WHERE item_id NOT IN ({placeholders})",  # noqa: S608
                [item.item_id for item in items],
            )
            connection.executemany(
                """
                INSERT INTO market_items (
                    item_id, name, description, category, image_path, water_price,
                    health_gain, satiety_gain, energy_gain, morale_gain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(item_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    category = excluded.category,
                    image_path = excluded.image_path,
                    water_price = excluded.water_price,
                    health_gain = excluded.health_gain,
                    satiety_gain = excluded.satiety_gain,
                    energy_gain = excluded.energy_gain,
                    morale_gain = excluded.morale_gain
                """,
                [
                    (
                        item.item_id,
                        item.name,
                        item.description,
                        item.category.value,
                        item.image_path,
                        item.water_price,
                        item.health_gain,
                        item.satiety_gain,
                        item.energy_gain,
                        item.morale_gain,
                    )
                    for item in items
                ],
            )
            connection.commit()

    @staticmethod
    def clean_profile_name(name: str) -> tuple[str, str]:
        display_name = " ".join(unicodedata.normalize("NFKC", name or "").split())
        if not display_name:
            raise ValueError("请输入存档名称。")
        if len(display_name) > 32:
            raise ValueError("存档名称最多 32 个字符。")
        if any(unicodedata.category(char).startswith("C") for char in display_name):
            raise ValueError("存档名称不能包含控制字符。")
        return display_name, display_name.casefold()

    def list_profiles(self) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name FROM profiles ORDER BY updated_at DESC, name ASC"
            ).fetchall()
        return [str(row["name"]) for row in rows]

    def get_active_profile_name(self) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM app_meta WHERE key = 'active_profile'"
            ).fetchone()
        return str(row["value"]) if row else None

    def set_active_profile_name(self, name: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_meta(key, value) VALUES ('active_profile', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (name,),
            )
            connection.commit()

    def create_profile(self, name: str) -> GameState:
        display_name, normalized_name = self.clean_profile_name(name)
        state = GameState(
            profile_name=display_name,
            market_item_ids=self.sample_market_ids(),
        )
        now = datetime.now(UTC).isoformat()
        payload = state.model_dump_json(exclude_computed_fields=True)
        try:
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO profiles(
                        name, normalized_name, state_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (display_name, normalized_name, payload, now, now),
                )
                connection.execute(
                    """
                    INSERT INTO app_meta(key, value) VALUES ('active_profile', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (display_name,),
                )
                connection.commit()
        except AppError as exc:
            if "UNIQUE constraint failed" in exc.detail:
                raise ValueError("这个存档名称已经存在。") from exc
            raise
        return state

    def load_profile(self, name: str) -> GameState:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT state_json FROM profiles WHERE name = ?",
                (name,),
            ).fetchone()
        if row is None:
            raise ValueError("所选存档不存在，请刷新存档列表。")
        try:
            return GameState.model_validate_json(str(row["state_json"]))
        except ValueError as exc:
            raise AppError(
                code=ErrorCode.INTERNAL,
                stage=PipelineStage.DATABASE,
                user_message="这个存档已损坏，未对它进行覆盖。",
                detail=str(exc),
            ) from exc

    def save_profile(self, state: GameState) -> None:
        display_name, normalized_name = self.clean_profile_name(state.profile_name)
        state.profile_name = display_name
        # Revalidate the whole snapshot before persistence. Pydantic assignment
        # validation cannot intercept in-place list mutations such as append().
        validated = GameState.model_validate(
            state.model_dump(exclude_computed_fields=True)
        )
        now = datetime.now(UTC).isoformat()
        payload = validated.model_dump_json(exclude_computed_fields=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE profiles
                SET normalized_name = ?, state_json = ?, updated_at = ?
                WHERE name = ?
                """,
                (normalized_name, payload, now, display_name),
            )
            if cursor.rowcount != 1:
                raise ValueError("当前存档不存在，无法保存。")
            connection.execute(
                """
                INSERT INTO app_meta(key, value) VALUES ('active_profile', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (display_name,),
            )
            connection.commit()

    def restart_profile(self, name: str) -> GameState:
        current = self.load_profile(name)
        fresh = GameState(
            profile_name=current.profile_name,
            market_item_ids=self.sample_market_ids(),
        )
        self.save_profile(fresh)
        return fresh

    def delete_profile(self, name: str) -> GameState:
        display_name, _ = self.clean_profile_name(name)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT name, state_json
                FROM profiles
                ORDER BY updated_at DESC, name ASC
                """
            ).fetchall()
            names = [str(row["name"]) for row in rows]
            if display_name not in names:
                raise ValueError("要删除的存档不存在，请刷新存档列表。")
            if len(names) <= 1:
                raise ValueError("至少需要保留一个存档，不能删除最后一个存档。")

            active_row = connection.execute(
                "SELECT value FROM app_meta WHERE key = 'active_profile'"
            ).fetchone()
            active_name = str(active_row["value"]) if active_row else None
            survivors = [profile for profile in names if profile != display_name]
            next_name = (
                active_name
                if active_name in survivors
                else survivors[0]
            )
            successor = next(row for row in rows if str(row["name"]) == next_name)
            try:
                next_state = GameState.model_validate_json(str(successor["state_json"]))
            except ValueError as exc:
                raise AppError(
                    code=ErrorCode.INTERNAL,
                    stage=PipelineStage.DATABASE,
                    user_message="备用存档已损坏，本次删除已取消。",
                    detail=str(exc),
                ) from exc

            connection.execute(
                "DELETE FROM profiles WHERE name = ?",
                (display_name,),
            )
            connection.execute(
                """
                INSERT INTO app_meta(key, value) VALUES ('active_profile', ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (next_name,),
            )
            connection.commit()
        return next_state

    def load_or_create_initial_profile(self) -> GameState:
        profiles = self.list_profiles()
        if not profiles:
            return self.create_profile("幸存者A")
        active = self.get_active_profile_name()
        if active in profiles:
            return self.load_profile(active)
        fallback = self.load_profile(profiles[0])
        self.set_active_profile_name(fallback.profile_name)
        return fallback

    def sample_market_ids(self, count: int = 5) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT item_id FROM market_items ORDER BY RANDOM() LIMIT ?",
                (count,),
            ).fetchall()
        if len(rows) != count:
            raise AppError(
                code=ErrorCode.INTERNAL,
                stage=PipelineStage.DATABASE,
                user_message="市场目录不完整，暂时无法刷新。",
                detail=f"requested={count}, found={len(rows)}",
            )
        return [str(row["item_id"]) for row in rows]

    def get_market_items(self, item_ids: Iterable[str]) -> list[MarketItem]:
        ordered_ids = list(dict.fromkeys(item_ids))
        if not ordered_ids:
            return []
        placeholders = ",".join("?" for _ in ordered_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM market_items WHERE item_id IN ({placeholders})",  # noqa: S608
                ordered_ids,
            ).fetchall()
        by_id = {
            str(row["item_id"]): MarketItem(
                item_id=str(row["item_id"]),
                name=str(row["name"]),
                description=str(row["description"]),
                category=str(row["category"]),
                image_path=str(row["image_path"]),
                water_price=float(row["water_price"]),
                health_gain=int(row["health_gain"]),
                satiety_gain=int(row["satiety_gain"]),
                energy_gain=int(row["energy_gain"]),
                morale_gain=int(row["morale_gain"]),
            )
            for row in rows
        }
        return [by_id[item_id] for item_id in ordered_ids if item_id in by_id]

    def get_market_item(self, item_id: str) -> MarketItem | None:
        items = self.get_market_items([item_id])
        return items[0] if items else None
