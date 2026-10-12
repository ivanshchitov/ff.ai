## Purpose
Undo agent operations while preserving user edits and the Git index.

## ADDED Requirements
### Requirement: Undo restores only the recorded agent change
The application SHALL implement the following contract: The /changes list and show commands SHALL expose record ids, statuses and diffs. Undo SHALL display a reverse diff and require confirmation, restore exact before content/existence/modes without changing the Git index, and remove only empty directories created by that change. Later manual edits or later applied changes on intersecting paths SHALL block the whole undo. Repeated undo SHALL make no writes. Interrupted undo SHALL recover conservatively from snapshots. Related task patches SHALL be marked rolled back and their task paused until explicit continuation.
#### Scenario: Authorized operation
- **WHEN** a valid request meets the stated preconditions
- **THEN** the existing facade performs the operation and reports its result
#### Scenario: Invalid or unavailable precondition
- **WHEN** validation, confirmation or a required resource is unavailable
- **THEN** the application reports the reason and preserves user data
