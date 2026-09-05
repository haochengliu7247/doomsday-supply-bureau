from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)


class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
    )


class ApocalypseScenario(StrEnum):
    CITY_BLACKOUT = "CITY_BLACKOUT"
    NUCLEAR_RUINS = "NUCLEAR_RUINS"
    FLOOD = "FLOOD"
    EXTREME_COLD = "EXTREME_COLD"
    EXTREME_HEAT = "EXTREME_HEAT"
    NATURE_RECLAIMS = "NATURE_RECLAIMS"

    @property
    def label(self) -> str:
        return {
            self.CITY_BLACKOUT: "城市断电",
            self.NUCLEAR_RUINS: "核灾废墟",
            self.FLOOD: "洪水",
            self.EXTREME_COLD: "极寒",
            self.EXTREME_HEAT: "高温 / 火灾",
            self.NATURE_RECLAIMS: "自然重占城市",
        }[self]


class ItemCategory(StrEnum):
    WATER = "water"
    FOOD = "food"
    ENERGY = "energy"
    ELECTRONICS = "electronics"
    TOOL = "tool"
    MEDICAL = "medical"
    SHELTER = "shelter"
    CLOTHING = "clothing"
    IDENTITY = "identity"
    MORALE = "morale"
    HAZARDOUS = "hazardous"
    OTHER = "other"


class Grade(StrEnum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    E = "E"
    F = "F"


class RecommendedAction(StrEnum):
    CARRY_NOW = "立即携带"
    CARRY = "建议携带"
    CARRY_IF_SPACE = "有空位再带"
    USE_HERE = "原地使用"
    TRADE = "建议交换"
    ABANDON = "建议放弃"
    NEVER_TOUCH = "绝对不要碰"


class EffectLevel(StrEnum):
    NONE = "none"
    TINY_POSITIVE = "tiny_positive"
    SMALL_POSITIVE = "small_positive"
    MEDIUM_POSITIVE = "medium_positive"
    LARGE_POSITIVE = "large_positive"
    TINY_NEGATIVE = "tiny_negative"
    SMALL_NEGATIVE = "small_negative"
    MEDIUM_NEGATIVE = "medium_negative"
    LARGE_NEGATIVE = "large_negative"


class PipelineStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"


class MarketItemCategory(StrEnum):
    FOOD = "food"
    MEDICAL = "medical"
    SURVIVAL = "survival"
    BOOK = "book"
    TOY = "toy"
    JUNK = "junk"


class EvidenceItem(StrictModel):
    observation: str = Field(min_length=1, max_length=300)
    inference: str = Field(min_length=1, max_length=300)
    confidence: float = Field(ge=0, le=1, strict=True)


class ItemStats(StrictModel):
    survival: int = Field(ge=1, le=5, strict=True)
    scarcity: int = Field(ge=1, le=5, strict=True)
    trade: int = Field(ge=1, le=5, strict=True)
    storage: int = Field(ge=1, le=5, strict=True)
    versatility: int = Field(ge=1, le=5, strict=True)
    risk: int = Field(ge=1, le=5, strict=True)


class PlayerEffectProfile(StrictModel):
    health: EffectLevel
    energy: EffectLevel
    hunger: EffectLevel
    morale: EffectLevel


class AppraisalPayload(StrictModel):
    original_item: str = Field(min_length=1, max_length=120)
    object_description: str = Field(min_length=1, max_length=1000)
    observed_evidence: list[str] = Field(min_length=1, max_length=12)
    material_evidence: list[EvidenceItem] = Field(min_length=1, max_length=8)
    materials: list[str] = Field(min_length=1, max_length=8)
    dominant_colors: list[str] = Field(min_length=1, max_length=8)
    original_features: list[str] = Field(min_length=1, max_length=16)
    preserve_features: list[str] = Field(min_length=1, max_length=16)
    apocalypse_changes: list[str] = Field(min_length=1, max_length=16)
    avoid_changes: list[str] = Field(min_length=1, max_length=16)
    apocalypse_name: str = Field(
        min_length=1,
        max_length=120,
        description=(
            "A creative post-apocalyptic translated identity for this specific object. "
            "It must not repeat the original object name or the apocalypse scenario label."
        ),
    )
    category: ItemCategory
    stats: ItemStats
    grade: Grade
    market_value_liters: float = Field(ge=0, le=9999, strict=True)
    market_value_description: str = Field(min_length=1, max_length=500)
    hidden_use: str = Field(min_length=1, max_length=800)
    analysis: str = Field(min_length=1, max_length=2000)
    verdict: str = Field(min_length=1, max_length=300)
    recommended_action: RecommendedAction
    player_effect_profile: PlayerEffectProfile
    image_edit_prompt: str = Field(
        min_length=1,
        max_length=4000,
        description=(
            "A self-contained 40-70 word English FLUX image-edit description. "
            "Use positive wording for the same object's structural anchors and visible "
            "surface weathering; do not list absent feature concepts."
        ),
    )
    confidence: float = Field(ge=0, le=1, strict=True)

    @field_validator(
        "observed_evidence",
        "materials",
        "dominant_colors",
        "original_features",
        "preserve_features",
        "apocalypse_changes",
        "avoid_changes",
    )
    @classmethod
    def strip_required_strings(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value and value.strip()]
        if not cleaned:
            raise ValueError("at least one non-empty entry is required")
        return cleaned

class AppraisalResult(AppraisalPayload):
    schema_version: str = Field(default="1.0", pattern=r"^1\.0$")
    is_fallback: bool = False
    warnings: list[str] = Field(default_factory=list)

    @classmethod
    def safe_fallback(cls, reason: str) -> AppraisalResult:
        return cls(
            original_item="未知物品",
            object_description="模型输出未通过严格校验，当前内容不可作为正式鉴定。",
            observed_evidence=["分析失败"],
            material_evidence=[
                EvidenceItem(
                    observation="缺少可靠视觉证据",
                    inference="unknown",
                    confidence=0.0,
                )
            ],
            materials=["unknown"],
            dominant_colors=["unknown"],
            original_features=["unknown"],
            preserve_features=["保持原图不变"],
            apocalypse_changes=["未生成"],
            avoid_changes=["不得依据本结果修改原物身份"],
            apocalypse_name="未归档的旧文明遗物",
            category=ItemCategory.OTHER,
            stats=ItemStats(
                survival=1,
                scarcity=1,
                trade=1,
                storage=1,
                versatility=1,
                risk=5,
            ),
            grade=Grade.F,
            market_value_liters=0.0,
            market_value_description="鉴定失败，不提供交易参考。",
            hidden_use="未知",
            analysis="鉴定数据不足，请重新扫描。",
            verdict="未知比损坏更危险。",
            recommended_action=RecommendedAction.ABANDON,
            player_effect_profile=PlayerEffectProfile(
                health=EffectLevel.NONE,
                energy=EffectLevel.NONE,
                hunger=EffectLevel.NONE,
                morale=EffectLevel.NONE,
            ),
            image_edit_prompt=(
                "Preserve the source image exactly as provided. Retain the same object, pixels, "
                "structure, color, viewpoint, framing, lighting, shadows, and background because "
                "the analysis result is unavailable and no reliable edit can be specified."
            ),
            confidence=0.0,
            is_fallback=True,
            warnings=[reason],
        )


class ImageIdentityVerdict(StrictModel):
    same_physical_object: bool
    category_preserved: bool
    silhouette_and_proportions_preserved: bool
    camera_and_composition_preserved: bool
    functional_features_preserved: bool
    only_surface_condition_changed: bool
    post_apocalyptic_damage_clearly_visible: bool
    before_functional_features: list[str] = Field(max_length=20)
    after_functional_features: list[str] = Field(max_length=20)
    added_features: list[str] = Field(max_length=12)
    missing_features: list[str] = Field(max_length=12)
    moved_or_duplicated_features: list[str] = Field(max_length=12)
    issues: list[str] = Field(max_length=12)
    confidence: float = Field(ge=0, le=1, strict=True)

    def is_acceptable(self, minimum_confidence: float) -> bool:
        checks = (
            self.same_physical_object,
            self.category_preserved,
            self.silhouette_and_proportions_preserved,
            self.camera_and_composition_preserved,
            self.functional_features_preserved,
            self.only_surface_condition_changed,
            self.post_apocalyptic_damage_clearly_visible,
        )
        structural_differences = (
            self.added_features,
            self.missing_features,
            self.moved_or_duplicated_features,
        )
        return (
            all(checks)
            and not any(structural_differences)
            and self.confidence >= minimum_confidence
        )


class MarketItem(StrictModel):
    item_id: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9_]+$")
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    category: MarketItemCategory
    image_path: str = Field(min_length=1, max_length=500)
    water_price: float = Field(ge=0, le=999, strict=True)
    health_gain: int = Field(default=0, ge=0, le=100, strict=True)
    satiety_gain: int = Field(default=0, ge=0, le=100, strict=True)
    energy_gain: int = Field(default=0, ge=0, le=100, strict=True)
    morale_gain: int = Field(default=0, ge=0, le=100, strict=True)


class PurchasedMarketItem(StrictModel):
    inventory_id: str = Field(
        default_factory=lambda: f"MKT-{uuid4().hex[:8].upper()}"
    )
    item: MarketItem
    purchased_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PlayerState(StrictModel):
    health: int = Field(default=100, ge=0, le=100)
    energy: int = Field(default=80, ge=0, le=100)
    hunger: int = Field(default=20, ge=0, le=100)
    morale: int = Field(default=70, ge=0, le=100)
    water: float = Field(default=10.0, ge=0)
    day: int = Field(default=1, ge=1)
    actions_today: int = Field(default=0, ge=0, le=2)
    toxic_damage_forecast: int = Field(default=4, ge=1, le=12)
    morale_shake_anchor: int = Field(default=70, ge=0, le=100)

    @computed_field
    @property
    def is_game_over(self) -> bool:
        return self.health <= 0

    @computed_field
    @property
    def survival_score(self) -> float:
        return round(self.health * 0.45 + self.energy * 0.30 + self.morale * 0.25, 1)


class InventoryItem(StrictModel):
    item_id: str = Field(default_factory=lambda: f"DSB-{uuid4().hex[:6].upper()}")
    appraisal: AppraisalResult
    original_image: str | None = None
    apocalypse_image: str | None = None
    source_description: str = Field(default="", max_length=1000)
    scenario: ApocalypseScenario
    apocalypse_years: float = Field(ge=0.25, le=30)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    consumed: bool = False


class GameLogEntry(StrictModel):
    category: str
    message: str
    day: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class GameState(StrictModel):
    profile_name: str = Field(default="幸存者A", min_length=1, max_length=32)
    player: PlayerState = Field(default_factory=PlayerState)
    inventory: list[InventoryItem] = Field(default_factory=list, max_length=6)
    market_inventory: list[PurchasedMarketItem] = Field(
        default_factory=list,
        max_length=6,
    )
    pending_item: InventoryItem | None = None
    market_item_ids: list[str] = Field(default_factory=list, max_length=5)
    scanned_items: int = Field(default=0, ge=0)
    highest_grade: Grade | None = None
    current_event: str | None = None
    log: list[GameLogEntry] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_total_inventory_capacity(self) -> GameState:
        if self.inventory_slots_used > 6:
            raise ValueError("背包总容量不能超过 6 格。")
        return self

    @computed_field
    @property
    def inventory_slots_used(self) -> int:
        return len(self.inventory) + len(self.market_inventory)


class StateTransition(StrictModel):
    state: GameState
    messages: list[str] = Field(default_factory=list)
    succeeded: bool = False
    day_advanced: bool = False
    health_damage: int = Field(default=0, ge=0, le=100)
    morale_shake: bool = False


class ScanRequest(StrictModel):
    scenario: ApocalypseScenario
    apocalypse_years: float = Field(ge=0.25, le=30)
    description: str = Field(default="", max_length=1000)


class PipelineResult(StrictModel):
    status: PipelineStatus
    item: InventoryItem
    original_image: str | None = None
    apocalypse_image: str | None = None
    timings_ms: dict[str, int] = Field(default_factory=dict)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
