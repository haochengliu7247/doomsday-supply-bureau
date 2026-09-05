import sqlite3
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from src.config import PROJECT_ROOT, Settings
from src.data.market_catalog import MARKET_CATALOG, MARKET_CATALOG_BY_ID
from src.schemas import (
    ApocalypseScenario,
    AppraisalResult,
    InventoryItem,
    MarketItemCategory,
    PurchasedMarketItem,
)
from src.services.game_repository import GameRepository


@pytest.fixture
def repository(tmp_path: Path) -> GameRepository:
    result = GameRepository(f"sqlite:///{(tmp_path / 'game.db').as_posix()}")
    result.initialize(MARKET_CATALOG)
    return result


def test_catalog_has_exactly_twenty_unique_items_and_required_categories() -> None:
    assert len(MARKET_CATALOG) == 20
    assert len({item.item_id for item in MARKET_CATALOG}) == 20
    assert sum(item.category is MarketItemCategory.FOOD for item in MARKET_CATALOG) == 3
    assert sum(item.category is MarketItemCategory.MEDICAL for item in MARKET_CATALOG) == 3
    assert any(item.category is MarketItemCategory.BOOK for item in MARKET_CATALOG)
    assert any(item.category is MarketItemCategory.TOY for item in MARKET_CATALOG)
    assert all(Path(item.image_path).is_file() for item in MARKET_CATALOG)
    for item in MARKET_CATALOG:
        with Image.open(item.image_path) as image:
            image.verify()


def test_initialize_is_repeatable_and_seeds_catalog(repository: GameRepository) -> None:
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            """
            INSERT INTO market_items(
                item_id, name, description, category, image_path, water_price
            ) VALUES ('rogue_item', '旧商品', '不应保留', 'junk', 'missing.png', 0)
            """
        )
    repository.initialize(MARKET_CATALOG)

    with sqlite3.connect(repository.database_path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM market_items").fetchone()[0]
        rogue = connection.execute(
            "SELECT 1 FROM market_items WHERE item_id = 'rogue_item'"
        ).fetchone()

    assert count == 20
    assert rogue is None


def test_named_profiles_round_trip_and_remain_isolated(repository: GameRepository) -> None:
    profile_a = repository.create_profile("A 人")
    profile_b = repository.create_profile("B 人")
    profile_a.player.health = 41
    repository.save_profile(profile_a)

    assert set(repository.list_profiles()) == {"A 人", "B 人"}
    assert repository.load_profile("A 人").player.health == 41
    assert repository.load_profile("B 人").player.health == 100
    assert profile_b.market_item_ids != []


def test_profile_names_are_normalized_and_duplicates_are_rejected(
    repository: GameRepository,
) -> None:
    repository.create_profile("  A   人  ")

    with pytest.raises(ValueError, match="已经存在"):
        repository.create_profile("A 人")
    with pytest.raises(ValueError, match="请输入"):
        repository.create_profile("   ")
    with pytest.raises(ValueError, match="32"):
        repository.create_profile("A" * 33)


def test_parameterized_profile_name_cannot_modify_database(
    repository: GameRepository,
) -> None:
    dangerous_name = "x'); DROP TABLE market_items;--"
    repository.create_profile(dangerous_name)

    assert repository.load_profile(dangerous_name).profile_name == dangerous_name
    assert len(repository.get_market_items(repository.sample_market_ids())) == 5


def test_restart_keeps_name_but_resets_run_and_market(repository: GameRepository) -> None:
    state = repository.create_profile("重开测试")
    state.player.health = 7
    state.player.day = 12
    state.market_item_ids = state.market_item_ids[:2]
    state.market_inventory = [
        PurchasedMarketItem(item=MARKET_CATALOG_BY_ID["toy_robot"])
    ]
    repository.save_profile(state)

    restarted = repository.restart_profile("重开测试")

    assert restarted.profile_name == "重开测试"
    assert restarted.player.health == 100
    assert restarted.player.day == 1
    assert len(restarted.market_item_ids) == 5
    assert len(set(restarted.market_item_ids)) == 5
    assert restarted.inventory == []
    assert restarted.market_inventory == []
    assert repository.load_profile("重开测试") == restarted


def test_delete_profile_switches_to_a_remaining_profile(
    repository: GameRepository,
) -> None:
    first = repository.create_profile("先遣队")
    repository.create_profile("后援队")
    repository.set_active_profile_name(first.profile_name)

    active = repository.delete_profile(first.profile_name)

    assert active.profile_name == "后援队"
    assert repository.list_profiles() == ["后援队"]
    assert repository.get_active_profile_name() == "后援队"
    with pytest.raises(ValueError, match="不存在"):
        repository.load_profile("先遣队")


def test_delete_profile_refuses_to_remove_the_last_save(
    repository: GameRepository,
) -> None:
    only = repository.create_profile("最后生还者")

    with pytest.raises(ValueError, match="最后一个存档"):
        repository.delete_profile(only.profile_name)

    assert repository.load_profile(only.profile_name) == only
    assert repository.get_active_profile_name() == only.profile_name


def test_deleting_an_inactive_profile_keeps_the_active_profile(
    repository: GameRepository,
) -> None:
    active = repository.create_profile("活动档")
    inactive = repository.create_profile("备用档")
    repository.set_active_profile_name(active.profile_name)

    returned = repository.delete_profile(inactive.profile_name)

    assert returned.profile_name == active.profile_name
    assert repository.get_active_profile_name() == active.profile_name
    assert repository.list_profiles() == [active.profile_name]


def test_market_inventory_round_trips_with_image_effects_and_instance_id(
    repository: GameRepository,
) -> None:
    state = repository.create_profile("市场背包回环")
    purchased = PurchasedMarketItem(
        item=MARKET_CATALOG_BY_ID["medical_first_aid"]
    )
    state.market_inventory.append(purchased)

    repository.save_profile(state)
    loaded = repository.load_profile(state.profile_name)

    assert loaded.market_inventory == [purchased]
    assert loaded.market_inventory[0].item.image_path == purchased.item.image_path
    assert loaded.market_inventory[0].item.health_gain == 20


def test_save_revalidates_in_place_inventory_mutations(
    repository: GameRepository,
) -> None:
    state = repository.create_profile("非法容量")
    item = MARKET_CATALOG_BY_ID["toy_robot"]
    state.inventory = [
        InventoryItem(
            appraisal=AppraisalResult.safe_fallback("test"),
            scenario=ApocalypseScenario.CITY_BLACKOUT,
            apocalypse_years=3,
        )
    ]
    state.market_inventory = [PurchasedMarketItem(item=item) for _ in range(5)]
    state.market_inventory.append(PurchasedMarketItem(item=item))

    with pytest.raises(ValidationError, match="总容量"):
        repository.save_profile(state)


def test_sold_out_market_stays_empty_after_reloading_profile(
    repository: GameRepository,
) -> None:
    state = repository.create_profile("售罄测试")
    state.market_item_ids = []
    repository.save_profile(state)

    reloaded = repository.load_profile("售罄测试")

    assert reloaded.market_item_ids == []


def test_market_sampling_returns_five_unique_database_items(
    repository: GameRepository,
) -> None:
    item_ids = repository.sample_market_ids()
    items = repository.get_market_items(item_ids)

    assert len(item_ids) == 5
    assert len(set(item_ids)) == 5
    assert [item.item_id for item in items] == item_ids


def test_relative_sqlite_url_is_resolved_from_project_root() -> None:
    settings = Settings(database_url="sqlite:///data/custom.db")

    assert settings.database_url == (
        f"sqlite:///{(PROJECT_ROOT / 'data' / 'custom.db').resolve().as_posix()}"
    )
