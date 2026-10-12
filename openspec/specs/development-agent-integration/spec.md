# development-agent-integration Specification

## Purpose
Connect confirmed development, recovery and undo across the existing TUI and web interfaces.

## Requirements

### Requirement: Recover and display the complete development lifecycle
The application SHALL implement the following contract: TUI and web SHALL run ordinary development requests through the existing pipeline and confirmed change journal, display operation ids, current evidence, check results and undo outcomes with their task, and allow confirmation responses without deadlock. Journal entries SHALL persist task and subtask identity before apply; restart SHALL reconcile applied and undone records into task state. Patch preparation SHALL reject unsupported file types and ambiguous paths before target writes, verify exact sandbox output paths and use regular UTF-8 snapshots only. Bounded read-only MCP evidence SHALL reach model context with provenance. Stopped request ids SHALL remain durable and SHALL never launch again on replay. Refusal, timeout, conflict and unavailable build/model SHALL be reported accurately while preserving user and staged files. Preparation SHALL not implement challenge deliverables.
#### Scenario: Authorized operation
- **WHEN** a valid request meets the stated preconditions
- **THEN** the existing facade performs the operation and reports its result
#### Scenario: Invalid or unavailable precondition
- **WHEN** validation, confirmation or a required resource is unavailable
- **THEN** the application reports the reason and preserves user data

#### Scenario: Crash after apply before saving task state
- **WHEN** a journal entry is applied but its TaskPatch was not saved before termination
- **THEN** startup restores the task/subtask association without rerunning its patch and undo updates that task
#### Scenario: Ambiguous Git path or nonregular mode
- **WHEN** patch headers have ambiguous unquoted whitespace or a nonregular numeric mode including leading zeroes
- **THEN** preparation refuses before target writes and never reads external symlink targets
#### Scenario: Header-like hunk content
- **WHEN** changed text begins with two plus or minus characters
- **THEN** hunk line counts distinguish content from path headers and the journal tracks the actual file
#### Scenario: Current read evidence with a large prior patch
- **WHEN** a later stage has a large previous diff and fresh read-only MCP constraints
- **THEN** a reserved bounded portion of context preserves new evidence with its source
#### Scenario: Lazy web read discovery
- **WHEN** a web development request needs tools and no discovery has run
- **THEN** declared stdio and HTTP servers are discovered and only explicitly read-only tools enter the automatic catalog
#### Scenario: Replay while confirming or after stopping
- **WHEN** a request_id is retried during confirmation or after task stop and restart
- **THEN** the previous task number is returned without a new launch and confirmation remains independently answerable
#### Scenario: Undo while runner is idle
- **WHEN** the user confirms undo after task completion
- **THEN** a new task snapshot immediately updates the original message and sidebar with rolled_back state and pause
