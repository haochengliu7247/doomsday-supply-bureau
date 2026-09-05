import pytest
from pydantic import ValidationError

from src.data.market_catalog import MARKET_CATALOG_BY_ID
from src.providers.mock_provider import MockVLMProvider
from src.schemas import (
    ApocalypseScenario,
    AppraisalPayload,
    AppraisalResult,
    GameState,
    ImageIdentityVerdict,
    InventoryItem,
    ItemStats,
    PurchasedMarketItem,
    ScanRequest,
)
from src.ui.gradio_app import _hydrate_state, _state_payload


def test_strict_models_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        GameState.model_validate({"unexpected": True})


def test_ratings_stay_between_one_and_five() -> None:
    with pytest.raises(ValidationError):
        ItemStats(survival=6)
    with pytest.raises(ValidationError):
        ItemStats(risk=0)


def test_identity_acceptance_is_determined_by_code() -> None:
    verdict = ImageIdentityVerdict(
        same_physical_object=True,
        category_preserved=True,
        silhouette_and_proportions_preserved=True,
        camera_and_composition_preserved=True,
        functional_features_preserved=True,
        only_surface_condition_changed=True,
        post_apocalyptic_damage_clearly_visible=True,
        before_functional_features=[],
        after_functional_features=[],
        added_features=[],
        missing_features=[],
        moved_or_duplicated_features=[],
        issues=[],
        confidence=0.95,
    )

    assert verdict.is_acceptable(0.8)
    verdict.post_apocalyptic_damage_clearly_visible = False
    assert not verdict.is_acceptable(0.8)
    verdict.post_apocalyptic_damage_clearly_visible = True
    verdict.added_features = ["new port"]
    assert not verdict.is_acceptable(0.8)


def test_combined_inventory_capacity_is_six() -> None:
    request = ScanRequest(
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=3,
        description="测试物资",
    )
    appraisal = MockVLMProvider().analyze(None, request)
    appraised_items = [
        InventoryItem(
            appraisal=appraisal.model_copy(deep=True),
            original_image="before.png",
            apocalypse_image="after.png",
            scenario=request.scenario,
            apocalypse_years=request.apocalypse_years,
        )
        for _ in range(6)
    ]
    market_item = PurchasedMarketItem(
        item=MARKET_CATALOG_BY_ID["toy_robot"]
    )

    valid = GameState(inventory=appraised_items[:5], market_inventory=[market_item])
    assert valid.inventory_slots_used == 6
    with pytest.raises(ValidationError, match="总容量"):
        GameState(inventory=appraised_items, market_inventory=[market_item])


def test_purchased_market_items_get_unique_instance_ids() -> None:
    item = MARKET_CATALOG_BY_ID["toy_robot"]

    first = PurchasedMarketItem(item=item)
    second = PurchasedMarketItem(item=item)

    assert first.inventory_id != second.inventory_id


def test_ratings_reject_numeric_strings() -> None:
    with pytest.raises(ValidationError):
        ItemStats(
            survival="5",
            scarcity=1,
            trade=1,
            storage=1,
            versatility=1,
            risk=1,
        )


def test_non_finite_water_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GameState.model_validate({"player": {"water": float("inf")}})


def test_model_payload_requires_all_business_fields() -> None:
    with pytest.raises(ValidationError):
        AppraisalPayload.model_validate({"original_item": "充电宝"})


def test_model_payload_rejects_blank_evidence() -> None:
    appraisal = MockVLMProvider().analyze(
        None,
        ScanRequest(
            scenario=ApocalypseScenario.CITY_BLACKOUT,
            apocalypse_years=3,
            description="充电宝",
        ),
    )
    payload = appraisal.model_dump(
        exclude={"schema_version", "is_fallback", "warnings"},
    )
    payload["observed_evidence"] = [" "]
    with pytest.raises(ValidationError):
        AppraisalPayload.model_validate(payload)


def test_fallback_is_conservative() -> None:
    fallback = AppraisalResult.safe_fallback("模型输出不可解析")
    assert fallback.market_value_liters == 0
    assert fallback.confidence == 0
    assert fallback.is_fallback
    assert fallback.grade.value == "F"
    assert "模型输出不可解析" in fallback.warnings


def test_gradio_state_payload_round_trips_without_computed_fields() -> None:
    state = GameState()
    payload = _state_payload(state)
    assert "inventory_slots_used" not in payload
    assert "is_game_over" not in payload["player"]
    assert "survival_score" not in payload["player"]
    assert _hydrate_state(payload) == state
