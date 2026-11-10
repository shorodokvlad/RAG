"""Turn Spring products and reviews into retrievable chunks (see PLAN.md §2).

Rules:
- Every chunk starts with the product name, so spec and review chunks still match
  queries that name the product (cheap "contextual retrieval").
- No prices or stock in chunks: those change constantly and are always read live from Spring.
- No reviewer names or ids: reviews are indexed without PII.
"""

import hashlib
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from app.indexing.specs import SpecItem, SpecSection, parse_description
from app.models import Product, Review

SourceType = Literal["product_card", "product_spec", "product_desc", "review", "faq"]

DESC_MAX_WORDS = 250
DESC_OVERLAP_WORDS = 40
REVIEW_MAX_WORDS = 250
SPEC_SECTION_MAX_ITEMS = 25
CARD_MAX_SPEC_ITEMS = 12
CARD_MAX_VARIANT_TITLES = 10
CARD_DESC_CHARS = 300

_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass(frozen=True)
class Chunk:
    source_type: SourceType
    source_id: int
    chunk_index: int
    content: str
    product_id: int | None = None
    category_id: int | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.source_type, self.source_id, self.chunk_index)

    @property
    def content_hash(self) -> str:
        """Covers what gets embedded (content only); metadata changes don't need a re-embed."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def metadata_json(self) -> str:
        return json.dumps(self.metadata, sort_keys=True, ensure_ascii=False, default=str)


def normalize_text(text: str | None) -> str:
    """Drop control characters and collapse runs of spaces; keep line breaks."""
    if not text:
        return ""
    text = _CONTROL.sub("", text)
    lines = (_WHITESPACE.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(line for line in lines if line)


def split_words(text: str, max_words: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [" ".join(words)] if words else []
    step = max(max_words - overlap, 1)
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start : start + max_words]))
        if start + max_words >= len(words):
            break
    return pieces


def _format_items(items: Iterable[SpecItem]) -> str:
    return "\n".join(f"{item.label}: {item.value}" for item in items)


def _variant_attributes(product: Product) -> dict[str, list[str]]:
    """{"Color": ["Black", "Blue"], "Storage": ["128GB"]}, keeping first-seen order."""
    merged: dict[str, list[str]] = {}
    for variant in product.variants:
        for name, value in variant.attributes.items():
            name, value = name.strip(), (value or "").strip()
            if name and value and value not in merged.setdefault(name, []):
                merged[name].append(value)
    return merged


def _base_metadata(product: Product) -> dict:
    metadata: dict = {"product_name": product.name}
    if product.category_name:
        metadata["category"] = product.category_name
    return metadata


def build_product_card(product: Product, sections: Sequence[SpecSection], prose: str | None) -> Chunk:
    lines = [product.name]
    if product.category_name:
        lines.append(f"Category: {product.category_name}")

    titles = [v.title.strip() for v in product.variants if v.title and v.title.strip()]
    if titles:
        lines.append("Options: " + ", ".join(titles[:CARD_MAX_VARIANT_TITLES]))

    attributes = _variant_attributes(product)
    if attributes:
        lines.append("Attributes: " + "; ".join(f"{k}: {', '.join(v)}" for k, v in attributes.items()))

    key_specs = [item for section in sections for item in section.items][:CARD_MAX_SPEC_ITEMS]
    if key_specs:
        lines.append("Key specs: " + "; ".join(f"{i.label}: {i.value}" for i in key_specs))

    if prose:
        flat = " ".join(prose.split())
        lines.append(flat[:CARD_DESC_CHARS] + ("…" if len(flat) > CARD_DESC_CHARS else ""))

    metadata = _base_metadata(product) | {
        "variant_count": len(product.variants),
        "review_count": product.review_count,
    }
    if product.average_rating is not None:
        metadata["average_rating"] = float(product.average_rating)

    return Chunk(
        source_type="product_card",
        source_id=product.id,
        chunk_index=0,
        content="\n".join(lines),
        product_id=product.id,
        category_id=product.category_id,
        metadata=metadata,
    )


def build_spec_chunks(product: Product, sections: Sequence[SpecSection]) -> list[Chunk]:
    chunks = []
    for section in sections:
        for start in range(0, len(section.items), SPEC_SECTION_MAX_ITEMS):
            part = section.items[start : start + SPEC_SECTION_MAX_ITEMS]
            chunks.append(
                Chunk(
                    source_type="product_spec",
                    source_id=product.id,
                    chunk_index=len(chunks),
                    content=f"{product.name} — {section.title}\n{_format_items(part)}",
                    product_id=product.id,
                    category_id=product.category_id,
                    metadata=_base_metadata(product) | {"section": section.title},
                )
            )
    return chunks


def build_description_chunks(product: Product, prose: str | None) -> list[Chunk]:
    if not prose:
        return []
    return [
        Chunk(
            source_type="product_desc",
            source_id=product.id,
            chunk_index=index,
            content=f"{product.name} — description\n{piece}",
            product_id=product.id,
            category_id=product.category_id,
            metadata=_base_metadata(product),
        )
        for index, piece in enumerate(split_words(prose, DESC_MAX_WORDS, DESC_OVERLAP_WORDS))
    ]


def build_review_chunks(product: Product, reviews: Iterable[Review]) -> list[Chunk]:
    chunks = []
    for review in reviews:
        text = normalize_text(review.content)
        if not text:
            continue
        metadata = _base_metadata(product) | {"rating": review.rating}
        if review.created_at:
            metadata["created_at"] = review.created_at.isoformat()
        for index, piece in enumerate(split_words(text, REVIEW_MAX_WORDS, 0)):
            chunks.append(
                Chunk(
                    source_type="review",
                    source_id=review.id,
                    chunk_index=index,
                    content=f"Review of {product.name} ({review.rating}/5):\n{piece}",
                    product_id=product.id,
                    category_id=product.category_id,
                    metadata=metadata,
                )
            )
    return chunks


def build_product_chunks(product: Product, reviews: Iterable[Review] = ()) -> list[Chunk]:
    """All chunks for one product, in a stable order."""
    parsed = parse_description(product.description)
    prose = normalize_text(parsed.prose) or None
    return [
        build_product_card(product, parsed.sections, prose),
        *build_spec_chunks(product, parsed.sections),
        *build_description_chunks(product, prose),
        *build_review_chunks(product, reviews),
    ]
