## Why
Natural chat requests use the existing development pipeline.

## What Changes
- Semantic classification SHALL distinguish answer, investigate, develop and clarify with strict JSON and counted usage. Errors, unknown kinds or missing goals SHALL clarify without launching a task. Explicit no-write requests SHALL override an inferred development intent. Ordinary questions and direct ask SHALL remain read-only. Development SHALL create one task and signal the existing runner; a busy or awaiting task SHALL be preserved. Clarification SHALL retain its original request and never approve an unseen patch. TUI and /api/ask SHALL share submit; answer fields SHALL remain compatible. An optional persisted request_id SHALL replay the same task without a new launch during execution, after completion and after restart; identical text with a different id SHALL be a new goal. Explicit /task SHALL bypass classification.

## Capabilities
### New Capabilities
- `chat-development-routing`: Natural chat requests use the existing development pipeline.
### Modified Capabilities

## Impact
Core, existing session facade, UI contracts, tests and documentation.
