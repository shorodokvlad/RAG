"""Store knowledge base: kb/*.md → FAQ chunks (returns, delivery, payment, ...).

One file is one document (its first "# " heading is the title); each "## " section is one
passage, prefixed with the document title so it still matches on its own.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path

from app.indexing.chunkers import Chunk, normalize_text

# Relative to the working directory (the repo root locally, /srv in the container).
KB_DIR = Path(os.environ.get("KB_DIR", "kb"))
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SKIPPED_FILES = {"readme.md"}


@dataclass(frozen=True)
class KbSection:
    heading: str
    body: str


@dataclass(frozen=True)
class KbDocument:
    slug: str
    title: str
    body: str

    @property
    def sections(self) -> list[KbSection]:
        return split_sections(self.body)


def parse_document(slug: str, text: str) -> KbDocument:
    text = normalize_text(text)
    title = slug.replace("-", " ").capitalize()
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.startswith("# "):
            title = line[2:].strip() or title
            lines = lines[:i] + lines[i + 1 :]
            break
    return KbDocument(slug=slug, title=title, body="\n".join(lines).strip())


def split_sections(body: str) -> list[KbSection]:
    sections: list[KbSection] = []
    heading = ""
    buffer: list[str] = []

    def flush() -> None:
        text = "\n".join(buffer).strip()
        if text:
            sections.append(KbSection(heading, text))

    for line in body.split("\n"):
        if line.startswith("## "):
            flush()
            heading, buffer = line[3:].strip(), []
        else:
            buffer.append(line)
    flush()
    return sections


def load_documents(directory: Path = KB_DIR) -> list[KbDocument]:
    if not directory.is_dir():
        return []
    documents = []
    for path in sorted(directory.glob("*.md")):
        slug = path.stem.lower()
        if path.name.lower() in SKIPPED_FILES or not _SLUG.match(slug):
            continue
        documents.append(parse_document(slug, path.read_text(encoding="utf-8")))
    return documents


def build_faq_chunks(document_id: int, document: KbDocument) -> list[Chunk]:
    chunks = []
    for index, section in enumerate(document.sections):
        title = f"{document.title}: {section.heading}" if section.heading else document.title
        chunks.append(
            Chunk(
                source_type="faq",
                source_id=document_id,
                chunk_index=index,
                content=f"{title}\n{section.body}",
                metadata={"slug": document.slug, "title": document.title, "heading": section.heading},
            )
        )
    return chunks
