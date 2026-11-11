"""Tools the chat agent can call. Each returns text for the model plus UI events.

The model only ever sees tool results as data inside <tool_result> text; product cards,
comparison tables and add-to-cart buttons go to the browser as structured events, so the
UI shows live, exact data no matter how the model words its answer.
"""

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Annotated, Any
from xml.sax.saxutils import escape

from pydantic import BaseModel, Field, ValidationError

from app.agent.events import Event
from app.features.common import product_facts
from app.features.search import ProductResult, SearchFilters, lowest_price, stock_state, to_result
from app.indexing.specs import parse_description
from app.models import Product

log = logging.getLogger(__name__)

MAX_COMPARE = 4
COMPARE_MAX_ROWS = 18
REVIEW_EXCERPTS = 6

ProductId = Annotated[int, Field(ge=1)]


# --- argument models (also the JSON schemas the model sees) --------------------------------


class SearchProductsArgs(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=200,
        description="What the shopper wants, in their words, without the price limits "
        "(e.g. 'phone with a great camera', 'apple watch for diving').",
    )
    max_price: Decimal | None = Field(default=None, ge=0, description="Highest price in EUR, if given.")
    min_price: Decimal | None = Field(default=None, ge=0, description="Lowest price in EUR, if given.")
    in_stock_only: bool = Field(default=False, description="Only products currently in stock.")


class ProductArgs(BaseModel):
    product_id: ProductId = Field(description="Product id from a previous tool result or the page context.")


class ReviewArgs(BaseModel):
    product_id: ProductId
    aspect: str | None = Field(
        default=None,
        max_length=100,
        description="Optional topic to look for in reviews, e.g. 'battery', 'camera', 'comfort'.",
    )


class CompareArgs(BaseModel):
    product_ids: list[ProductId] = Field(
        min_length=2, max_length=MAX_COMPARE, description="2 to 4 product ids."
    )


class StoreInfoArgs(BaseModel):
    question: str = Field(min_length=2, max_length=200, description="The shopper's question about the store.")


class AddToCartArgs(BaseModel):
    product_id: ProductId
    variant_id: int | None = Field(
        default=None,
        description="Id of the chosen option (color/storage), required when the product has options.",
    )
    quantity: int = Field(default=1, ge=1, le=10)


@dataclass
class ToolResult:
    text: str
    events: list[Event] = field(default_factory=list)
    product_ids: list[int] = field(default_factory=list)


@dataclass
class ToolContext:
    """What tools may use for one chat turn. user_id comes from Spring's JWT check, never from chat text."""

    services: Any
    user_id: int | None = None


# --- helpers -------------------------------------------------------------------------------


def _result_tag(tool: str, body: str) -> str:
    return f'<tool_result tool="{tool}">\n{body}\n</tool_result>'


def euros(value: Decimal | None) -> str:
    """€899, €899.50: how prices appear on the site (no "€899.0")."""
    if value is None:
        return "—"
    amount = Decimal(value).quantize(Decimal("0.01"))
    return f"€{amount:.0f}" if amount == amount.to_integral_value() else f"€{amount:.2f}"


def _card(result: ProductResult) -> dict:
    return result.model_dump(mode="json")


def _product_line(r: ProductResult) -> str:
    price = euros(r.price) if r.price is not None else "price unknown"
    stock = {True: "in stock", False: "out of stock", None: "stock unknown"}[r.in_stock]
    rating = f", rated {r.average_rating}/5 ({r.review_count} reviews)" if r.review_count else ""
    category = f" [{r.category_name}]" if r.category_name else ""
    line = f"- #{r.id} {escape(r.name)}{category}: from {price}, {stock}{rating}"
    return line + (f"\n  matched: {escape(r.match)}" if r.match else "")


def _variant_lines(product: Product) -> list[str]:
    lines = []
    for v in product.variants[:20]:
        price = euros(v.price) if v.price is not None else "base price"
        stock = (
            "" if v.stock_quantity is None else (", in stock" if v.stock_quantity > 0 else ", out of stock")
        )
        label = v.title or ", ".join(f"{k}: {val}" for k, val in v.attributes.items()) or f"option {v.id}"
        lines.append(f"- option id {v.id}: {escape(label)}: {price}{stock}")
    return lines


async def _product(ctx: ToolContext, product_id: int) -> Product | None:
    return await ctx.services.spring.get_product(product_id)


def _not_found(tool: str, product_id: int) -> ToolResult:
    return ToolResult(_result_tag(tool, f"No product with id {product_id} exists."))


# --- tools ---------------------------------------------------------------------------------


async def search_products(ctx: ToolContext, args: SearchProductsArgs) -> ToolResult:
    filters = SearchFilters(
        min_price=args.min_price, max_price=args.max_price, in_stock_only=args.in_stock_only
    )
    results = await ctx.services.searcher.search(args.query, filters, limit=6)
    if not results:
        return ToolResult(_result_tag("search_products", "No matching products found."))
    body = "Products shown to the shopper as cards, best match first:\n" + "\n".join(
        map(_product_line, results)
    )
    return ToolResult(
        _result_tag("search_products", body),
        [Event("products", {"items": [_card(r) for r in results]})],
        [r.id for r in results],
    )


async def similar_products(ctx: ToolContext, args: ProductArgs) -> ToolResult:
    results = await ctx.services.searcher.similar(args.product_id, limit=6)
    if not results:
        return ToolResult(_result_tag("similar_products", "No similar products found."))
    body = "Similar products shown as cards:\n" + "\n".join(map(_product_line, results))
    return ToolResult(
        _result_tag("similar_products", body),
        [Event("products", {"items": [_card(r) for r in results]})],
        [r.id for r in results],
    )


async def get_product_details(ctx: ToolContext, args: ProductArgs) -> ToolResult:
    product = await _product(ctx, args.product_id)
    if product is None:
        return _not_found("get_product_details", args.product_id)
    result = to_result(product)
    lines = [
        _product_line(result),
        "Options (live prices and stock):",
        *(_variant_lines(product) or ["- none"]),
    ]
    facts = product_facts(product)
    lines.append(facts if facts else "No detailed specifications are available for this product.")
    return ToolResult(
        _result_tag("get_product_details", "\n".join(lines)),
        [Event("products", {"items": [_card(result)]})],
        [product.id],
    )


async def get_review_insights(ctx: ToolContext, args: ReviewArgs) -> ToolResult:
    services = ctx.services
    product = await _product(ctx, args.product_id)
    if product is None:
        return _not_found("get_review_insights", args.product_id)
    reviews = await services.spring.get_reviews(product.id)
    if not reviews:
        return ToolResult(_result_tag("get_review_insights", f"{escape(product.name)} has no reviews yet."))

    lines = [f"Reviews of #{product.id} {escape(product.name)}: {len(reviews)} in total."]
    record = None
    if services.review_summaries is not None:
        try:
            record = await services.review_summaries.get_or_generate(product, reviews)
        except Exception:  # the excerpts below still answer the question
            log.warning("review summary unavailable for product %s", product.id, exc_info=True)
    if record is not None:
        s = record.summary
        lines.append(f"Average rating {s.average_rating}/5. Overall: {escape(s.verdict)}")
        lines.extend(f"- liked ({p.count} reviews): {escape(p.text)}" for p in s.pros)
        lines.extend(f"- disliked ({c.count} reviews): {escape(c.text)}" for c in s.cons)

    by_id = {r.id: r for r in reviews}
    excerpts = []
    if args.aspect:
        excerpts = [
            by_id[i] for i in await services.searcher.hybrid.review_ids(product.id, args.aspect) if i in by_id
        ]
    if not excerpts and record is None:
        # Newest first; reviews without a date go last.
        excerpts = sorted(reviews, key=lambda r: (r.created_at is not None, r.created_at or 0), reverse=True)
    for review in excerpts[:REVIEW_EXCERPTS]:
        text = " ".join((review.content or "").split())[:400]
        lines.append(f'<review rating="{review.rating}">{escape(text)}</review>')
    return ToolResult(_result_tag("get_review_insights", "\n".join(lines)), product_ids=[product.id])


def comparison_table(products: list[Product]) -> dict:
    """Spec table aligned by label (case-insensitive); shared specs first, then the rest."""
    specs: list[dict[str, str]] = []
    labels: dict[str, str] = {}  # lowercase → display label, first seen wins
    for product in products:
        values: dict[str, str] = {}
        for section in parse_description(product.description).sections:
            for item in section.items:
                key = item.label.strip().lower()
                labels.setdefault(key, item.label.strip())
                values.setdefault(key, item.value)
        specs.append(values)

    def coverage(key: str) -> int:
        return sum(key in values for values in specs)

    ordered = sorted(labels, key=lambda k: (-coverage(k), list(labels).index(k)))
    ordered = [k for k in ordered if coverage(k) >= min(2, len(products))][:COMPARE_MAX_ROWS]

    results = [to_result(p) for p in products]
    rows = [
        {"label": "Price", "values": [euros(r.price) for r in results]},
        {
            "label": "Rating",
            "values": [
                f"{r.average_rating}/5 ({r.review_count})" if r.review_count else "—" for r in results
            ],
        },
        {
            "label": "Stock",
            "values": [{True: "In stock", False: "Out of stock", None: "—"}[r.in_stock] for r in results],
        },
    ]
    rows += [{"label": labels[k], "values": [values.get(k, "—") for values in specs]} for k in ordered]
    return {"products": [_card(r) for r in results], "rows": rows}


async def compare_products(ctx: ToolContext, args: CompareArgs) -> ToolResult:
    ids = list(dict.fromkeys(args.product_ids))
    found = {p.id: p for p in await ctx.services.spring.get_products_by_ids(ids)}
    products = [found[i] for i in ids if i in found]
    missing = [i for i in ids if i not in found]
    if len(products) < 2:
        return ToolResult(_result_tag("compare_products", f"Can't compare: products not found: {missing}."))
    table = comparison_table(products)
    names = " | ".join(f"#{p.id} {escape(p.name)}" for p in products)
    lines = [f"Comparison table shown to the shopper. Columns: {names}"]
    lines.extend(
        f"- {escape(row['label'])}: " + " | ".join(escape(v) for v in row["values"]) for row in table["rows"]
    )
    if missing:
        lines.append(f"Not found: {missing}")
    return ToolResult(
        _result_tag("compare_products", "\n".join(lines)),
        [Event("comparison", table)],
        [p.id for p in products],
    )


async def search_store_info(ctx: ToolContext, args: StoreInfoArgs) -> ToolResult:
    passages = await ctx.services.searcher.hybrid.search_kb(args.question, limit=4)
    if not passages:
        return ToolResult(_result_tag("search_store_info", "No store information found on this topic."))
    body = "\n".join(f'<document title="{escape(p.title)}">{escape(p.content)}</document>' for p in passages)
    sources = [{"title": p.title, "slug": p.slug} for p in passages]
    return ToolResult(_result_tag("search_store_info", body), [Event("sources", {"items": sources})])


async def suggest_add_to_cart(ctx: ToolContext, args: AddToCartArgs) -> ToolResult:
    product = await _product(ctx, args.product_id)
    if product is None:
        return _not_found("suggest_add_to_cart", args.product_id)
    variant = None
    if product.variants:
        variant = next((v for v in product.variants if v.id == args.variant_id), None)
        if variant is None:
            options = "\n".join(_variant_lines(product))
            return ToolResult(
                _result_tag(
                    "suggest_add_to_cart",
                    "No button shown: this product has options. Ask the shopper which one they want, "
                    f"then call again with its variant_id.\n{options}",
                )
            )
        if variant.stock_quantity is not None and variant.stock_quantity <= 0:
            return ToolResult(
                _result_tag("suggest_add_to_cart", "No button shown: that option is out of stock.")
            )
    elif stock_state(product) is False:
        return ToolResult(_result_tag("suggest_add_to_cart", "No button shown: the product is out of stock."))

    price = variant.price if variant is not None and variant.price is not None else lowest_price(product)
    action = {
        "type": "add_to_cart",
        "product": _card(to_result(product)),
        "variant": (
            {"id": variant.id, "title": variant.title, "attributes": variant.attributes, "price": price}
            if variant is not None
            else None
        ),
        "quantity": args.quantity,
    }
    label = f"{product.name} ({variant.title})" if variant is not None and variant.title else product.name
    return ToolResult(
        _result_tag(
            "suggest_add_to_cart",
            f"An 'Add to cart' button for {escape(label)} at {euros(price)} is shown. "
            "The shopper must press it; nothing was added yet.",
        ),
        [Event("action", action)],
        [product.id],
    )


# --- registry ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args: type[BaseModel]
    run: Callable[[ToolContext, Any], Awaitable[ToolResult]]

    def schema(self) -> dict:
        parameters = self.args.model_json_schema()
        parameters.pop("title", None)
        for prop in parameters.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "search_products",
            "Find products in the store catalog by what the shopper needs. Shows product cards.",
            SearchProductsArgs,
            search_products,
        ),
        Tool(
            "get_product_details",
            "Live price, stock, options and full specifications of one product.",
            ProductArgs,
            get_product_details,
        ),
        Tool(
            "get_review_insights",
            "What customers say about a product: summary of reviews and review excerpts, optionally "
            "about one aspect.",
            ReviewArgs,
            get_review_insights,
        ),
        Tool(
            "compare_products",
            "Compare 2-4 products side by side. Shows a comparison table.",
            CompareArgs,
            compare_products,
        ),
        Tool(
            "similar_products",
            "Products similar to a given product (alternatives). Shows product cards.",
            ProductArgs,
            similar_products,
        ),
        Tool(
            "search_store_info",
            "Store policies and help: delivery, returns, warranty, payment, ordering, contact.",
            StoreInfoArgs,
            search_store_info,
        ),
        Tool(
            "suggest_add_to_cart",
            "Show the shopper an 'Add to cart' button for a product option. Use only when they ask "
            "to buy or add something. It doesn't add anything by itself.",
            AddToCartArgs,
            suggest_add_to_cart,
        ),
    ]
}


def tool_schemas() -> list[dict]:
    return [t.schema() for t in TOOLS.values()]


async def run_tool(ctx: ToolContext, name: str, arguments: str) -> ToolResult:
    """Run a tool call from the model. Bad calls become error text the model can correct."""
    tool = TOOLS.get(name)
    if tool is None:
        return ToolResult(_result_tag(name, f"Error: unknown tool '{escape(name)}'."))
    try:
        args = tool.args.model_validate(json.loads(arguments or "{}"))
    except (json.JSONDecodeError, ValidationError) as exc:
        detail = str(exc).splitlines()[0][:300]
        return ToolResult(
            _result_tag(name, f"Error: invalid arguments ({escape(detail)}). Fix them and retry.")
        )
    try:
        return await tool.run(ctx, args)
    except Exception:
        log.exception("tool %s failed", name)
        return ToolResult(_result_tag(name, "Error: this information is unavailable right now."))
