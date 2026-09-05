from src.data.common_items import (
    COMMON_ITEMS_EXPECTED_COUNT,
    load_common_item_lexicon,
)


def test_common_item_lexicon_has_exactly_one_thousand_valid_terms() -> None:
    lexicon = load_common_item_lexicon()
    rows = lexicon.manifest_rows()

    assert lexicon.version == "common-items-zh-v1"
    assert len(lexicon.categories) == 16
    assert len(rows) == COMMON_ITEMS_EXPECTED_COUNT
    assert [ordinal for ordinal, _, _ in rows] == list(
        range(1, COMMON_ITEMS_EXPECTED_COUNT + 1)
    )
    assert len(lexicon.alias_map()) == 490
    assert all(category.quota == len(category.items) for category in lexicon.categories)
