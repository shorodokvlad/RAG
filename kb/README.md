# Store knowledge base

Markdown documents the chat assistant answers store questions from (returns, delivery,
payment, ordering, contact). They're indexed into `ai.chunks` as `source_type='faq'`:

- one document per file; the first `# ` heading is its title,
- each `## ` section becomes one retrievable passage, so keep a section about one question,
- the file name is the document's id (`returns.md` → `returns`); renaming it re-indexes it.

Changes are picked up when the service starts, on the nightly reconcile, or right away with
`POST /index/kb`. Only edited sections are re-embedded. This README isn't indexed.

Keep these documents true: the assistant repeats them to customers word for word.
