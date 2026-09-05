from __future__ import annotations

import unicodedata
from pathlib import Path

from pydantic import Field, model_validator

from src.schemas import StrictModel

COMMON_ITEMS_PATH = Path(__file__).with_name("common_items_zh_v1.json")
COMMON_ITEMS_EXPECTED_COUNT = 1000


def _normalized_term(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split()).casefold()


class CommonItem(StrictModel):
    canonical: str = Field(min_length=1, max_length=40)
    aliases: list[str] = Field(default_factory=list, max_length=12)


class CommonItemCategory(StrictModel):
    id: str = Field(min_length=1, max_length=60, pattern=r"^[a-z0-9_]+$")
    label: str = Field(min_length=1, max_length=40)
    quota: int = Field(ge=1, le=200)
    items: list[CommonItem] = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def quota_matches_items(self) -> CommonItemCategory:
        if self.quota != len(self.items):
            raise ValueError(f"category {self.id} quota does not match item count")
        return self


class CommonItemLexicon(StrictModel):
    version: str = Field(min_length=1, max_length=80, pattern=r"^[a-z0-9-]+$")
    categories: list[CommonItemCategory] = Field(min_length=1, max_length=40)

    @model_validator(mode="after")
    def validate_unique_terms(self) -> CommonItemLexicon:
        canonical_terms: dict[str, str] = {}
        alias_targets: dict[str, str] = {}
        for category in self.categories:
            for item in category.items:
                canonical = _normalized_term(item.canonical)
                if canonical in canonical_terms:
                    raise ValueError(f"duplicate canonical term: {item.canonical}")
                canonical_terms[canonical] = item.canonical
                for raw_alias in item.aliases:
                    alias = _normalized_term(raw_alias)
                    if not alias or alias == canonical:
                        raise ValueError(f"invalid alias for {item.canonical}: {raw_alias}")
                    previous = alias_targets.get(alias)
                    if previous is not None and previous != canonical:
                        raise ValueError(f"alias maps to multiple terms: {raw_alias}")
                    if alias in canonical_terms and alias != canonical:
                        raise ValueError(f"alias is another canonical term: {raw_alias}")
                    alias_targets[alias] = canonical
        overlap = set(alias_targets).intersection(canonical_terms)
        if overlap:
            raise ValueError(
                f"aliases overlap canonical terms: {', '.join(sorted(overlap)[:5])}"
            )
        if len(canonical_terms) != COMMON_ITEMS_EXPECTED_COUNT:
            raise ValueError(
                f"common item lexicon must contain exactly {COMMON_ITEMS_EXPECTED_COUNT} terms"
            )
        return self

    def manifest_rows(self) -> list[tuple[int, str, str]]:
        rows: list[tuple[int, str, str]] = []
        for category in self.categories:
            for item in category.items:
                rows.append((len(rows) + 1, category.id, item.canonical))
        return rows

    def alias_map(self) -> dict[str, str]:
        return {
            alias: item.canonical
            for category in self.categories
            for item in category.items
            for alias in item.aliases
        }


def load_common_item_lexicon(
    path: Path = COMMON_ITEMS_PATH,
) -> CommonItemLexicon:
    return CommonItemLexicon.model_validate_json(path.read_text(encoding="utf-8"))
