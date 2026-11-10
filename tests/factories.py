import json
from datetime import datetime

from app.models import Product, Review, Variant

SPEC_JSON = json.dumps(
    [
        {
            "title": "GENERAL CHARACTERISTICS",
            "items": [
                {"label": "Operating System", "value": "iOS"},
                {"label": "Connectivity", "value": "5G"},
                {"label": "Sensors", "value": ""},
            ],
        },
        {"title": "DISPLAY", "items": [{"label": "Screen Size", "value": "6.1 inch"}]},
        {"title": "EMPTY", "items": [{"label": "Weight", "value": "  "}]},
    ]
)


def make_product(**overrides) -> Product:
    data = {
        "id": 12,
        "name": "iPhone 16",
        "description": SPEC_JSON,
        "category_id": 3,
        "category_name": "Phones",
        "price": "4999.99",
        "stock_quantity": 7,
        "average_rating": "4.5",
        "review_count": 2,
        "variants": [
            Variant(id=1, title="Black / 128GB", attributes={"Color": "Black", "Storage": "128GB"}),
            Variant(id=2, title="Blue / 256GB", attributes={"Color": "Blue", "Storage": "256GB"}),
            Variant(id=3, title="Black / 256GB", attributes={"Color": "Black", "Storage": "256GB"}),
        ],
    }
    data.update(overrides)
    return Product(**data)


def make_review(review_id: int = 100, rating: int = 5, content: str = "Great battery life") -> Review:
    return Review(
        id=review_id, product_id=12, rating=rating, content=content, created_at=datetime(2026, 11, 1, 12, 0)
    )
