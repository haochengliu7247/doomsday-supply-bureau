import pytest

from src.data.market_catalog import MARKET_CATALOG_BY_ID
from src.game.engine import (
    abandon_inventory_item,
    abandon_market_inventory_item,
    abandon_pending_item,
    apply_inventory_item,
    apply_market_inventory_item,
    apply_pending_item,
    buy_market_item,
    can_scavenge,
    register_scanned_item,
    rest,
    sell_inventory_item,
    sell_pending_item,
    store_pending_item,
)
from src.schemas import (
    ApocalypseScenario,
    AppraisalResult,
    EffectLevel,
    GameState,
    Grade,
    InventoryItem,
    ItemCategory,
    PlayerEffectProfile,
    PurchasedMarketItem,
)


def make_item(
    name: str = "便携式应急工具",
    *,
    category: ItemCategory = ItemCategory.TOOL,
    years: float = 3,
    grade: Grade = Grade.B,
    profile: PlayerEffectProfile | None = None,
) -> InventoryItem:
    appraisal = AppraisalResult.safe_fallback("test fixture")
    appraisal.original_item = "测试物品"
    appraisal.apocalypse_name = name
    appraisal.category = category
    appraisal.grade = grade
    appraisal.is_fallback = False
    appraisal.warnings = []
    appraisal.player_effect_profile = profile or PlayerEffectProfile(
        health=EffectLevel.NONE,
        energy=EffectLevel.MEDIUM_POSITIVE,
        hunger=EffectLevel.MEDIUM_POSITIVE,
        morale=EffectLevel.MEDIUM_POSITIVE,
    )
    return InventoryItem(
        appraisal=appraisal,
        original_image="before.png",
        apocalypse_image="after.png",
        scenario=ApocalypseScenario.CITY_BLACKOUT,
        apocalypse_years=years,
    )


def test_initial_state_uses_positive_satiety_conversion_and_toxic_forecast() -> None:
    state = GameState()
    assert state.player.health == 100
    assert state.player.energy == 80
    assert state.player.hunger == 20
    assert 100 - state.player.hunger == 80
    assert state.player.morale == 70
    assert state.player.water == 10.0
    assert state.player.day == 1
    assert state.player.toxic_damage_forecast == 4


def test_scan_is_one_of_only_two_actions_that_advance_a_day() -> None:
    first = register_scanned_item(GameState(), make_item())

    assert first.succeeded is True
    assert first.day_advanced is True
    assert first.state.player.day == 2
    assert first.state.player.energy == 72
    assert first.state.player.hunger == 30
    assert first.state.player.morale == 68
    assert first.state.player.water == 9.5
    assert first.state.player.health == 96
    assert first.health_damage == 4
    assert first.state.scanned_items == 1


def test_scan_cannot_overwrite_an_unresolved_item() -> None:
    state = GameState(pending_item=make_item("尚未处理"))
    before = state.model_dump()

    result = register_scanned_item(state, make_item("新物资"))

    assert result.succeeded is False
    assert result.day_advanced is False
    assert result.state.model_dump() == before
    assert "待处理物资" in result.messages[0]


def test_rest_advances_one_day_without_reducing_satiety() -> None:
    state = GameState()
    state.player.energy = 0
    forecast = state.player.toxic_damage_forecast

    result = rest(state)

    assert result.succeeded is True
    assert result.day_advanced is True
    assert result.state.player.day == 2
    assert result.state.player.energy == 27
    assert result.state.player.hunger == 20
    assert result.state.player.water == 9.5
    assert result.state.player.morale == 69
    assert result.state.player.health == 100 - forecast
    assert "不会降低饱食度" in result.messages[0]


def test_toxic_damage_is_forecast_before_the_day_advances_and_then_changes() -> None:
    state = GameState()
    shown_forecast = state.player.toxic_damage_forecast

    result = rest(state)

    assert result.health_damage == shown_forecast
    assert 3 <= result.state.player.toxic_damage_forecast <= 7
    assert result.state.player.toxic_damage_forecast != shown_forecast
    assert any("下一天预计损失" in message for message in result.messages)


def test_water_shortage_and_toxic_damage_both_reduce_health() -> None:
    state = GameState()
    state.player.water = 0.2

    result = rest(state)

    assert result.state.player.water == 0
    assert result.state.player.health == 86
    assert result.health_damage == 14
    assert any("净水不足" in message for message in result.messages)


def test_non_food_use_does_not_restore_satiety_or_advance_time() -> None:
    state = GameState(pending_item=make_item(category=ItemCategory.TOOL))
    before_hunger = state.player.hunger

    result = apply_pending_item(state)

    assert result.succeeded is True
    assert result.day_advanced is False
    assert result.state.player.day == 1
    assert result.state.player.hunger == before_hunger
    assert "非食物不补充饱食度" in result.messages[0]


def test_generated_food_slowly_restores_satiety_but_old_food_costs_health() -> None:
    state = GameState(pending_item=make_item(category=ItemCategory.FOOD, years=3))
    state.player.hunger = 50

    result = apply_pending_item(state)

    assert result.state.player.day == 1
    assert result.state.player.hunger == 42
    assert 100 - result.state.player.hunger == 58
    assert result.state.player.health == 95
    assert result.health_damage == 5
    assert "经历 3 年" in result.messages[0]


def test_three_month_generated_food_has_no_age_health_cost() -> None:
    state = GameState(pending_item=make_item(category=ItemCategory.FOOD, years=0.25))

    result = apply_pending_item(state)

    assert result.state.player.health == 100
    assert result.state.player.hunger == 12


def test_market_food_enters_backpack_and_only_applies_when_used() -> None:
    item = MARKET_CATALOG_BY_ID["food_compressed_ration"]
    state = GameState(market_item_ids=[item.item_id])
    state.player.hunger = 80

    purchased = buy_market_item(state, item)

    assert purchased.succeeded is True
    assert purchased.day_advanced is False
    assert purchased.state.player.day == 1
    assert purchased.state.player.hunger == 80
    assert purchased.state.player.health == 100
    assert purchased.state.player.water == 6.5
    assert item.item_id not in purchased.state.market_item_ids
    assert len(purchased.state.market_inventory) == 1
    stored = purchased.state.market_inventory[0]
    assert stored.item.item_id == item.item_id

    used = apply_market_inventory_item(purchased.state, stored.inventory_id)

    assert used.succeeded is True
    assert used.day_advanced is False
    assert used.state.player.day == 1
    assert used.state.player.hunger == 35
    assert used.state.player.water == 6.5
    assert used.state.market_inventory == []


def test_market_medical_supply_restores_health_only_after_backpack_use() -> None:
    item = MARKET_CATALOG_BY_ID["medical_first_aid"]
    state = GameState(market_item_ids=[item.item_id])
    state.player.health = 50

    purchased = buy_market_item(state, item)

    assert purchased.state.player.health == 50
    assert purchased.state.player.water == 6.0

    stored = purchased.state.market_inventory[0]
    used = apply_market_inventory_item(purchased.state, stored.inventory_id)

    assert used.state.player.health == 70
    assert used.state.player.water == 6.0
    assert used.state.market_inventory == []


def test_market_purchase_rejects_full_combined_backpack_without_mutation() -> None:
    item = MARKET_CATALOG_BY_ID["food_compressed_ration"]
    state = GameState(
        inventory=[make_item(f"鉴定物资 {index}") for index in range(5)],
        market_inventory=[PurchasedMarketItem(item=item)],
        market_item_ids=[item.item_id],
    )
    before = state.model_dump()

    result = buy_market_item(state, item)

    assert result.succeeded is False
    assert result.state.model_dump() == before
    assert "背包已经装满" in result.messages[0]


def test_market_inventory_can_be_abandoned_but_not_resold() -> None:
    item = MARKET_CATALOG_BY_ID["toy_robot"]
    purchased = PurchasedMarketItem(item=item)
    state = GameState(market_inventory=[purchased])
    before = state.model_dump()

    sold = sell_inventory_item(state, purchased.inventory_id)
    abandoned = abandon_market_inventory_item(state, purchased.inventory_id)

    assert sold.succeeded is False
    assert sold.state.model_dump() == before
    assert "不能转售" in sold.messages[0]
    assert abandoned.succeeded is True
    assert abandoned.day_advanced is False
    assert abandoned.state.market_inventory == []


def test_zero_effect_market_item_is_consumed_without_changing_stats() -> None:
    item = MARKET_CATALOG_BY_ID["book_novel"]
    purchased = PurchasedMarketItem(item=item)
    state = GameState(market_inventory=[purchased])
    before_player = state.player.model_dump()

    result = apply_market_inventory_item(state, purchased.inventory_id)

    assert result.succeeded is True
    assert result.state.player.model_dump() == before_player
    assert result.state.market_inventory == []
    assert "生命 +0" in result.messages[0]


def test_duplicate_market_purchases_get_unique_inventory_ids() -> None:
    item = MARKET_CATALOG_BY_ID["toy_robot"]
    state = GameState(market_item_ids=[item.item_id])

    first = buy_market_item(state, item)
    first.state.market_item_ids = [item.item_id]
    second = buy_market_item(first.state, item)

    inventory_ids = [entry.inventory_id for entry in second.state.market_inventory]
    assert len(inventory_ids) == 2
    assert len(set(inventory_ids)) == 2


def test_market_rejects_unaffordable_or_stale_selection_without_mutation() -> None:
    item = MARKET_CATALOG_BY_ID["food_canned_meal"]
    state = GameState(market_item_ids=[item.item_id])
    state.player.water = 0.5
    before = state.model_dump()

    unaffordable = buy_market_item(state, item)
    stale = buy_market_item(state, MARKET_CATALOG_BY_ID["toy_robot"])

    assert unaffordable.succeeded is False
    assert unaffordable.state.model_dump() == before
    assert "净水不足" in unaffordable.messages[0]
    assert stale.succeeded is False
    assert stale.state.model_dump() == before


def test_backpack_actions_free_slot_without_advancing_time() -> None:
    first = make_item("第一件")
    second = make_item("第二件")
    state = GameState(inventory=[first, second])

    used = apply_inventory_item(state, first.item_id)
    abandoned = abandon_inventory_item(state, second.item_id)

    assert used.state.player.day == 1
    assert [item.item_id for item in used.state.inventory] == [second.item_id]
    assert abandoned.state.player.day == 1
    assert [item.item_id for item in abandoned.state.inventory] == [first.item_id]


def test_store_pending_item_respects_combined_backpack_capacity() -> None:
    market_item = MARKET_CATALOG_BY_ID["toy_chess"]
    state = GameState(
        inventory=[make_item(f"物资 {index}") for index in range(5)],
        market_inventory=[PurchasedMarketItem(item=market_item)],
        pending_item=make_item("待入库物资"),
    )
    before = state.model_dump()

    result = store_pending_item(state)

    assert result.succeeded is False
    assert result.state.model_dump() == before
    assert "背包已经装满" in result.messages[0]


def test_store_abandon_and_sale_do_not_advance_time() -> None:
    stored = store_pending_item(GameState(pending_item=make_item()))
    assert stored.state.player.day == 1
    assert stored.state.inventory_slots_used == 1

    sold = sell_inventory_item(stored.state, stored.state.inventory[0].item_id)
    assert sold.state.player.day == 1
    assert sold.state.inventory == []
    assert sold.state.player.water == pytest.approx(12.0)

    abandoned = abandon_pending_item(GameState(pending_item=make_item("待放弃物资")))
    assert abandoned.state.player.day == 1
    assert abandoned.state.pending_item is None


@pytest.mark.parametrize(
    ("grade", "water_gain"),
    [
        (Grade.D, 0.1),
        (Grade.C, 0.5),
        (Grade.B, 2.0),
        (Grade.A, 5.0),
        (Grade.S, 15.0),
    ],
)
def test_d_or_better_sales_gain_fixed_water_by_grade(
    grade: Grade,
    water_gain: float,
) -> None:
    state = GameState(pending_item=make_item(grade=grade))

    result = sell_pending_item(state)

    assert result.state.player.water == pytest.approx(10.0 + water_gain)
    assert result.state.player.morale == 70


@pytest.mark.parametrize(("grade", "morale_gain"), [(Grade.E, 3), (Grade.F, 2)])
def test_e_and_f_sales_gain_morale_instead_of_water(
    grade: Grade,
    morale_gain: int,
) -> None:
    state = GameState(pending_item=make_item(grade=grade))

    result = sell_pending_item(state)

    assert result.state.player.water == 10.0
    assert result.state.player.morale == 70 + morale_gain


def test_every_accumulated_five_morale_loss_triggers_a_shake() -> None:
    state = GameState()
    for index in range(2):
        scanned = register_scanned_item(state, make_item(f"物资{index}"))
        assert scanned.morale_shake is False
        state = abandon_pending_item(scanned.state).state

    third = register_scanned_item(state, make_item("第三件"))

    assert third.state.player.morale == 64
    assert third.morale_shake is True


def test_invalid_backpack_selection_does_not_change_state() -> None:
    state = GameState(inventory=[make_item()])
    before = state.model_dump()

    result = apply_inventory_item(state, "missing")

    assert result.succeeded is False
    assert result.state.model_dump() == before


def test_game_over_blocks_all_state_actions() -> None:
    item = MARKET_CATALOG_BY_ID["food_compressed_ration"]
    market_entry = PurchasedMarketItem(item=item)
    state = GameState(
        pending_item=make_item(),
        inventory=[make_item("背包物资")],
        market_inventory=[market_entry],
        market_item_ids=[item.item_id],
    )
    state.player.health = 0
    before = state.model_dump()

    transitions = [
        rest(state),
        store_pending_item(state),
        apply_pending_item(state),
        abandon_pending_item(state),
        sell_pending_item(state),
        apply_inventory_item(state, state.inventory[0].item_id),
        sell_inventory_item(state, state.inventory[0].item_id),
        abandon_inventory_item(state, state.inventory[0].item_id),
        apply_market_inventory_item(state, market_entry.inventory_id),
        abandon_market_inventory_item(state, market_entry.inventory_id),
        buy_market_item(state, item),
    ]

    assert all(transition.state.model_dump() == before for transition in transitions)
    assert all(transition.succeeded is False for transition in transitions)


def test_cannot_scavenge_at_zero_energy() -> None:
    state = GameState()
    state.player.energy = 0
    allowed, reason = can_scavenge(state)
    assert not allowed
    assert "精疲力尽" in reason
