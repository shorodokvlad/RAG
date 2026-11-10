"""Python port of `client/src/utils/specParser.js#parseSpecifications`.

Spring stores a product's specifications inside `Product.description`, either as a JSON
array `[{"title": "...", "items": [{"label": "...", "value": "..."}]}]` (written by the
admin SpecBuilder) or as legacy text ("Label: value" / tab-separated lines with section
headers). Keep this in sync with the JS version so we index what the product page shows.
"""

import json
from dataclasses import dataclass

DEFAULT_SECTION = "GENERAL CHARACTERISTICS"
_SKIPPED_HEADERS = {"specificatii", "specificații", "specifications"}


@dataclass(frozen=True)
class SpecItem:
    label: str
    value: str


@dataclass(frozen=True)
class SpecSection:
    title: str
    items: tuple[SpecItem, ...]


@dataclass(frozen=True)
class ParsedDescription:
    sections: tuple[SpecSection, ...]
    # True when the description is the SpecBuilder JSON: then it holds no prose at all.
    is_structured: bool
    # Prose worth indexing as text (None for structured descriptions).
    prose: str | None


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ("" if value is None else str(value).strip())


def _parse_json(text: str) -> list[SpecSection] | None:
    try:
        parsed = json.loads(text)
    except ValueError:
        return None
    if not isinstance(parsed, list) or not parsed:
        return None

    sections = []
    for raw in parsed:
        if not isinstance(raw, dict):
            continue
        items = tuple(
            SpecItem(_clean(item.get("label")), _clean(item.get("value")))
            for item in raw.get("items") or []
            if isinstance(item, dict) and _clean(item.get("label")) and _clean(item.get("value"))
        )
        if items:
            sections.append(SpecSection(_clean(raw.get("title")) or "SPECIFICATIONS", items))
    return sections


def _parse_text(text: str) -> list[SpecSection] | None:
    sections: list[SpecSection] = []
    title: str | None = None
    items: list[SpecItem] = []
    parsed_any_pair = False

    for line in (raw.strip() for raw in text.split("\n")):
        if not line or line.lower() in _SKIPPED_HEADERS:
            continue

        label = value = ""
        if "\t" in line:
            label, _, value = line.partition("\t")
        elif ": " in line:
            label, _, value = line.partition(": ")
        label, value = label.strip(), value.strip()

        if label and value:
            parsed_any_pair = True
            if title is None:
                title = DEFAULT_SECTION
            items.append(SpecItem(label, value))
        else:
            # Any other line starts a new section (and drops an empty previous one), as in the JS.
            if title is not None and items:
                sections.append(SpecSection(title, tuple(items)))
            title, items = line.upper(), []

    if title is not None and items:
        sections.append(SpecSection(title, tuple(items)))

    return sections if parsed_any_pair else None


def parse_specifications(text: str | None) -> list[SpecSection] | None:
    """Same contract as the JS: a list of sections, or None when there are no specs."""
    if not text or not text.strip():
        return None
    trimmed = text.strip()
    if trimmed.startswith("["):
        sections = _parse_json(trimmed)
        if sections is not None:
            return sections
    return _parse_text(trimmed)


def parse_description(text: str | None) -> ParsedDescription:
    """Split a description into spec sections and indexable prose.

    Legacy text can mix prose with "Label: value" lines (the product page then shows it as
    specs), so for text descriptions we also keep the raw text as prose: nothing is lost.
    """
    if not text or not text.strip():
        return ParsedDescription((), is_structured=False, prose=None)

    trimmed = text.strip()
    if trimmed.startswith("["):
        sections = _parse_json(trimmed)
        if sections is not None:
            return ParsedDescription(tuple(sections), is_structured=True, prose=None)

    sections = _parse_text(trimmed) or []
    return ParsedDescription(tuple(sections), is_structured=False, prose=trimmed)
