"""Index the catalog from the command line.

python -m app.indexing.backfill                    # full reindex + prune deleted products
python -m app.indexing.backfill --product-id 12    # one or more products
python -m app.indexing.backfill --dry-run          # print chunks; no database or API key needed
"""

import argparse
import asyncio
import logging
import sys

from app.clients.spring import SpringClient
from app.config import get_settings
from app.db import create_pool
from app.indexing.chunkers import build_product_chunks
from app.indexing.indexer import Indexer, reindex_all, reindex_product
from app.providers import build_embedder, missing_embedder_key


async def dry_run(spring: SpringClient, product_ids: list[int], show: int) -> None:
    async def products():
        if product_ids:
            for pid in product_ids:
                product = await spring.get_product(pid)
                if product is None:
                    print(f"product {pid}: not found")
                else:
                    yield product
        else:
            async for product in spring.iter_products():
                yield product

    total = 0
    async for product in products():
        chunks = build_product_chunks(product, await spring.get_reviews(product.id))
        total += len(chunks)
        counts: dict[str, int] = {}
        for chunk in chunks:
            counts[chunk.source_type] = counts.get(chunk.source_type, 0) + 1
        print(f"\n=== product {product.id}: {product.name} → {len(chunks)} chunks {counts}")
        for chunk in chunks[:show]:
            print(f"--- {chunk.source_type}#{chunk.source_id}.{chunk.chunk_index}\n{chunk.content}")
    print(f"\n{total} chunks in total (dry run, nothing written)")


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--product-id", type=int, action="append", default=[], help="repeatable")
    parser.add_argument("--dry-run", action="store_true", help="only build and print chunks")
    parser.add_argument("--show", type=int, default=3, help="chunks to print per product in --dry-run")
    parser.add_argument("--page-size", type=int, default=50)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    spring = SpringClient(
        settings.spring_base_url, settings.service_token.get_secret_value(), settings.spring_timeout_seconds
    )
    try:
        if args.dry_run:
            await dry_run(spring, args.product_id, args.show)
            return 0

        embedder = build_embedder(settings)
        if embedder is None:
            key = missing_embedder_key(settings)
            print(f"{key} is not set (use --dry-run to test without it)", file=sys.stderr)
            return 2

        pool = await create_pool(settings)
        try:
            indexer = Indexer(pool, embedder)
            if args.product_id:
                for pid in args.product_id:
                    await reindex_product(spring, indexer, pid)
                failed = []
            else:
                summary = await reindex_all(spring, indexer, args.page_size)
                failed = summary.failed
                print(summary)
            print(f"Embedding model: {settings.embedding_provider}/{embedder.model}")
        finally:
            await pool.close()
            await embedder.aclose()
        return 1 if failed else 0
    finally:
        await spring.aclose()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
