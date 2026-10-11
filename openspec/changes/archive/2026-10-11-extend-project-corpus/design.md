## Context
The domain defines extensions and glob exclusions. Existing window chunks work for Markdown.
## Goals / Non-Goals
Add local documentation; retain index schema and portal retriever.
## Decisions
Reuse scan/chunk/search and preserve useful JSON/YAML while excluding filenames and state directories. Keep Russian search terms alongside code identifiers.
## Risks / Trade-offs
Documents may be stale; retrieval citations retain file and lines.
