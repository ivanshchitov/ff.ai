## Context
Reuse the existing session and task pipeline.
## Goals / Non-Goals
Development stages read current project evidence. Preparation only, no challenge deliverables.
## Decisions
Planning and execution SHALL discover and read relevant repository files using existing retrieval and bounded read-only tool flows. Validation SHALL receive fresh task-owned changes and deterministic check/build results; unrelated working changes SHALL not be attributed to the agent. File conventions and build structure SHALL inform the plan as evidence, not permission. Applied/undone changes SHALL mark the index stale until successful rebuild, with direct current-file reads as fallback. Unknown write tools, command runners and schedulers SHALL not enter read-only flows, including ordinary ask().
## Risks / Trade-offs
Persist conservative failure states; conflicts require explicit user action.
