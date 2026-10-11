## ADDED Requirements
### Requirement: Development stages read current project evidence
The application SHALL implement the following contract: Planning and execution SHALL discover and read relevant repository files using existing retrieval and bounded read-only tool flows. Validation SHALL receive fresh task-owned changes and deterministic check/build results; unrelated working changes SHALL not be attributed to the agent. File conventions and build structure SHALL inform the plan as evidence, not permission. Applied/undone changes SHALL mark the index stale until successful rebuild, with direct current-file reads as fallback. Unknown write tools, command runners and schedulers SHALL not enter read-only flows, including ordinary ask().
#### Scenario: Authorized operation
- **WHEN** a valid request meets the stated preconditions
- **THEN** the existing facade performs the operation and reports its result
#### Scenario: Invalid or unavailable precondition
- **WHEN** validation, confirmation or a required resource is unavailable
- **THEN** the application reports the reason and preserves user data
