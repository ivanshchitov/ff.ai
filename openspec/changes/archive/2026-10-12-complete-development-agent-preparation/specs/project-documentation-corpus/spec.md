## Purpose
Make local project documentation searchable alongside source code with stable path and line references.

## ADDED Requirements
### Requirement: Local documentation shares the code corpus
The application SHALL index README and docs Markdown with verbatim line ranges using the existing window strategy. It SHALL retain useful JSON/YAML API descriptions while excluding environment secrets, IDE state, recordings and journals. Portal `/rag-docs` behavior SHALL remain unchanged.
#### Scenario: Mixed project sources
- **WHEN** a repository contains README.md, docs/design.md, API JSON/YAML and IDE/state files
- **THEN** documentation and API descriptions are searchable and IDE/state files are excluded
#### Scenario: Russian documentation
- **WHEN** a Russian question matches Russian Markdown content
- **THEN** retrieval can deliver that content with its original file path and lines
