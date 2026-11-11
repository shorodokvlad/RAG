"""F3: AI review summary: verdict, pros/cons and themes, each backed by real review ids.

Up to SINGLE_PASS_MAX reviews go to the model in one call. Above that it's map-reduce:
each batch is reduced to short aspect mentions, and the summary is written from those.
Every review id the model cites is checked against the real reviews, and counts are
computed here, never taken from the model.
"""

import json
import logging
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Literal
from xml.sax.saxutils import escape, quoteattr

import asyncpg
from pydantic import BaseModel, Field

from app.clients.llm import LLM
from app.features.common import UNTRUSTED_DATA_RULE, KeyedLocks, fingerprint
from app.indexing.chunkers import normalize_text
from app.models import Product, Review

log = logging.getLogger(__name__)

PROMPT_VERSION = "review-summary-v1"
SINGLE_PASS_MAX = 60
MAP_BATCH_SIZE = 40
MAX_POINTS = 6
MAX_REVIEW_CHARS = 1500

Sentiment = Literal["positive", "mixed", "negative"]


# --- what the model returns -------------------------------------------------------------


class DraftPoint(BaseModel):
    text: str = Field(description="Short phrase, e.g. 'Battery easily lasts two days'")
    review_ids: list[int] = Field(description="Ids of every review that supports this point")


class DraftTheme(BaseModel):
    name: str = Field(description="1-3 word aspect, e.g. 'Battery life', 'Build quality'")
    positive_review_ids: list[int]
    negative_review_ids: list[int]


class ReviewSummaryDraft(BaseModel):
    verdict: str = Field(description="1-2 sentences: what customers think overall")
    sentiment: Sentiment
    pros: list[DraftPoint] = Field(description="Up to 6 praised aspects, most mentioned first")
    cons: list[DraftPoint] = Field(description="Up to 6 criticised aspects, most mentioned first")
    themes: list[DraftTheme] = Field(description="Up to 6 aspects customers discuss most")


class AspectMention(BaseModel):
    aspect: str = Field(description="1-3 word aspect name in English, lower case")
    sentiment: Literal["positive", "negative", "neutral"]
    review_id: int
    quote: str = Field(description="At most 12 words copied from the review")


class AspectMentions(BaseModel):
    mentions: list[AspectMention]


# --- what we store and return -------------------------------------------------------------


class SummaryPoint(BaseModel):
    text: str
    count: int
    review_ids: list[int]


class SummaryTheme(BaseModel):
    name: str
    positive: int
    negative: int
    positive_review_ids: list[int]
    negative_review_ids: list[int]


class ReviewSummary(BaseModel):
    verdict: str
    sentiment: Sentiment
    pros: list[SummaryPoint]
    cons: list[SummaryPoint]
    themes: list[SummaryTheme]
    review_count: int
    average_rating: float
    rating_distribution: dict[str, int]


@dataclass(frozen=True)
class ReviewSummaryRecord:
    product_id: int
    summary: ReviewSummary
    model: str
    generated_at: datetime
    # True when reviews changed but regeneration is throttled (shown until the next refresh).
    stale: bool = False


SUMMARY_SYSTEM_PROMPT = f"""You summarize customer reviews for an online electronics store.

Rules:
- Use only what the reviews say. Don't add facts about the product from elsewhere.
- Every pro, con and theme must list the ids of the reviews that support it. Only cite
  reviews that actually say it.
- Merge points that mean the same thing. Prefer aspects mentioned by several customers;
  a point backed by one review is fine only when there are few reviews.
- The verdict is balanced and neutral; mention disagreement when customers disagree.
- No hype, no emojis, no quotes longer than a few words.
- {UNTRUSTED_DATA_RULE}"""

MAP_SYSTEM_PROMPT = f"""You extract what customers say about specific aspects of a product.

For each review, list the aspects it judges (battery, screen, delivery, value for money, ...),
whether it's positive, negative or neutral about each, and a short quote copied from it.
Skip reviews with no concrete opinion. {UNTRUSTED_DATA_RULE}"""


def _format_review(review: Review) -> str:
    text = escape(normalize_text(review.content)[:MAX_REVIEW_CHARS])
    attrs = f"id={quoteattr(str(review.id))} rating={quoteattr(str(review.rating))}"
    return f"<review {attrs}>{text}</review>"


def _usable(reviews: list[Review]) -> list[Review]:
    return [r for r in reviews if normalize_text(r.content)]


def _rating_stats(reviews: list[Review]) -> tuple[float, dict[str, int]]:
    distribution = {str(star): 0 for star in range(5, 0, -1)}
    for review in reviews:
        key = str(min(max(review.rating, 1), 5))
        distribution[key] += 1
    average = round(sum(r.rating for r in reviews) / len(reviews), 1) if reviews else 0.0
    return average, distribution


def _valid_ids(ids: list[int], allowed: set[int]) -> list[int]:
    seen: list[int] = []
    for review_id in ids:
        if review_id in allowed and review_id not in seen:
            seen.append(review_id)
    return seen


def verify_draft(draft: ReviewSummaryDraft, reviews: list[Review]) -> ReviewSummary:
    """Drop citations of reviews that don't exist; drop points left with no support."""
    allowed = {r.id for r in reviews}

    def points(raw: list[DraftPoint]) -> list[SummaryPoint]:
        out = []
        for point in raw:
            ids = _valid_ids(point.review_ids, allowed)
            if ids and point.text.strip():
                out.append(SummaryPoint(text=point.text.strip(), count=len(ids), review_ids=ids))
        out.sort(key=lambda p: p.count, reverse=True)
        return out[:MAX_POINTS]

    themes = []
    for theme in draft.themes:
        pos = _valid_ids(theme.positive_review_ids, allowed)
        neg = _valid_ids(theme.negative_review_ids, allowed)
        if (pos or neg) and theme.name.strip():
            themes.append(
                SummaryTheme(
                    name=theme.name.strip(),
                    positive=len(pos),
                    negative=len(neg),
                    positive_review_ids=pos,
                    negative_review_ids=neg,
                )
            )
    themes.sort(key=lambda t: t.positive + t.negative, reverse=True)

    average, distribution = _rating_stats(reviews)
    return ReviewSummary(
        verdict=draft.verdict.strip(),
        sentiment=draft.sentiment,
        pros=points(draft.pros),
        cons=points(draft.cons),
        themes=themes[:MAX_POINTS],
        review_count=len(reviews),
        average_rating=average,
        rating_distribution=distribution,
    )


class ReviewSummaryService:
    def __init__(
        self,
        pool: asyncpg.Pool,
        llm: LLM,
        language: str,
        min_reviews: int = 3,
        min_interval: timedelta = timedelta(minutes=10),
    ):
        self._pool = pool
        self._llm = llm
        self._language = language
        self.min_reviews = min_reviews
        self._min_interval = min_interval
        self._locks = KeyedLocks()

    def _reviews_hash(self, reviews: list[Review]) -> str:
        data = sorted((r.id, r.rating, normalize_text(r.content)) for r in reviews)
        return fingerprint(PROMPT_VERSION, self._llm.model, self._language, data)

    async def _load(self, product_id: int):
        async with self._pool.acquire() as conn:
            return await conn.fetchrow(
                "SELECT summary, reviews_hash, model, generated_at FROM ai.review_summaries "
                "WHERE product_id = $1",
                product_id,
            )

    async def get_or_generate(
        self, product: Product, reviews: list[Review], force: bool = False
    ) -> ReviewSummaryRecord | None:
        """None when there are too few reviews (any old summary is then removed)."""
        usable = _usable(reviews)
        if len(usable) < self.min_reviews:
            await self.delete(product.id)
            return None
        reviews_hash = self._reviews_hash(usable)

        async with self._locks(product.id):
            row = await self._load(product.id)
            if row is not None:
                record = ReviewSummaryRecord(
                    product.id,
                    ReviewSummary.model_validate_json(row["summary"]),
                    row["model"],
                    row["generated_at"],
                )
                if row["reviews_hash"] == reviews_hash:
                    return record
                recently = datetime.now(UTC) - row["generated_at"] < self._min_interval
                if recently and not force:
                    return replace(record, stale=True)

            summary, model = await self._generate(product, usable)
            async with self._pool.acquire() as conn:
                saved = await conn.fetchrow(
                    """
                    INSERT INTO ai.review_summaries (product_id, summary, review_count, reviews_hash, model)
                    VALUES ($1, $2::jsonb, $3, $4, $5)
                    ON CONFLICT (product_id) DO UPDATE SET
                        summary = EXCLUDED.summary, review_count = EXCLUDED.review_count,
                        reviews_hash = EXCLUDED.reviews_hash, model = EXCLUDED.model,
                        generated_at = now()
                    RETURNING generated_at
                    """,
                    product.id,
                    json.dumps(summary.model_dump(), ensure_ascii=False),
                    len(usable),
                    reviews_hash,
                    model,
                )
            log.info("generated review summary for product %s (%s reviews)", product.id, len(usable))
            return ReviewSummaryRecord(product.id, summary, model, saved["generated_at"])

    async def _generate(self, product: Product, reviews: list[Review]) -> tuple[ReviewSummary, str]:
        header = (
            f"Product: {escape(product.name)}\nNumber of reviews: {len(reviews)}\n"
            f"Write the verdict, pros, cons and theme names in {self._language}.\n\n"
        )
        if len(reviews) <= SINGLE_PASS_MAX:
            body = "\n".join(_format_review(r) for r in reviews)
        else:
            body = await self._map_mentions(product, reviews)

        result = await self._llm.parse(
            system=SUMMARY_SYSTEM_PROMPT,
            user=header + body,
            output=ReviewSummaryDraft,
            max_tokens=3000,
        )
        return verify_draft(result.output, reviews), result.model

    async def _map_mentions(self, product: Product, reviews: list[Review]) -> str:
        """Condense many reviews into aspect mentions (one LLM call per batch, sequential to
        stay within free-tier rate limits)."""
        lines = []
        for start in range(0, len(reviews), MAP_BATCH_SIZE):
            batch = reviews[start : start + MAP_BATCH_SIZE]
            batch_ids = {r.id for r in batch}
            result = await self._llm.parse(
                system=MAP_SYSTEM_PROMPT,
                user=f"Product: {escape(product.name)}\n\n" + "\n".join(_format_review(r) for r in batch),
                output=AspectMentions,
                max_tokens=4000,
            )
            for m in result.output.mentions:
                if m.review_id in batch_ids:
                    lines.append(
                        f"<mention review={quoteattr(str(m.review_id))} sentiment={quoteattr(m.sentiment)} "
                        f"aspect={quoteattr(m.aspect)}>{escape(m.quote)}</mention>"
                    )
        return "Aspect mentions extracted from the reviews:\n" + "\n".join(lines)

    async def delete(self, product_id: int) -> None:
        async with self._pool.acquire() as conn:
            await conn.execute("DELETE FROM ai.review_summaries WHERE product_id = $1", product_id)
