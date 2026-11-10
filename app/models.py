"""Shapes returned by Spring's /internal/ai/** endpoints (camelCase JSON)."""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class SpringModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, frozen=True)


class Variant(SpringModel):
    id: int
    title: str | None = None
    attributes: dict[str, str] = Field(default_factory=dict)
    price: Decimal | None = None
    stock_quantity: int | None = None


class Product(SpringModel):
    id: int
    name: str
    description: str | None = None
    category_id: int | None = None
    category_name: str | None = None
    price: Decimal | None = None
    stock_quantity: int | None = None
    image_url: str | None = None
    average_rating: Decimal | None = None
    review_count: int = 0
    variants: list[Variant] = Field(default_factory=list)


class Review(SpringModel):
    id: int
    product_id: int
    rating: int
    content: str | None = None
    created_at: datetime | None = None


class ProductPage(SpringModel):
    items: list[Product]
    page: int
    size: int
    total_pages: int
    total_elements: int
