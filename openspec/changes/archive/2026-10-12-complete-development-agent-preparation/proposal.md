## Why
Recover and display the complete development lifecycle.

## What Changes
- TUI and web SHALL run ordinary development requests through the existing pipeline and confirmed change journal, display operation ids, current evidence, check results and undo outcomes with their task, and allow confirmation responses without deadlock. Journal entries SHALL persist task and subtask identity before apply; restart SHALL reconcile applied and undone records into task state. Patch preparation SHALL reject unsupported file types and ambiguous paths before target writes, verify exact sandbox output paths and use regular UTF-8 snapshots only. Bounded read-only MCP evidence SHALL reach model context with provenance. Stopped request ids SHALL remain durable and SHALL never launch again on replay. Refusal, timeout, conflict and unavailable build/model SHALL be reported accurately while preserving user and staged files. Preparation SHALL not implement challenge deliverables.

## Capabilities
### New Capabilities
- `development-agent-integration`: Recover and display the complete development lifecycle.
### Modified Capabilities
- Regenerate the five preparation capabilities with explicit Purpose metadata; preserve every existing requirement and scenario.

## Impact
Core, existing session facade, UI contracts, tests and documentation.
