## ADDED Requirements
### Requirement: Confirmed changes have a durable project journal
The application SHALL implement the following contract: Text file creation, editing and deletion SHALL use one project-bound record with private before/after snapshots outside the project. Every patch header path SHALL stay inside the root and avoid symlinks and .git; binary and rename patches SHALL be rejected. Application SHALL require approval of the displayed diff, repeat fingerprints and domain checks after approval, leave the Git index unchanged and recover interrupted known states without overwriting unknown user edits. Standalone task execution SHALL use the same gate and deny writes without confirmation.
#### Scenario: Authorized operation
- **WHEN** a valid request meets the stated preconditions
- **THEN** the existing facade performs the operation and reports its result
#### Scenario: Invalid or unavailable precondition
- **WHEN** validation, confirmation or a required resource is unavailable
- **THEN** the application reports the reason and preserves user data

#### Scenario: User edits during confirmation
- **WHEN** a touched file changes while the displayed diff awaits approval
- **THEN** the whole package is refused with a conflict and the newer content remains intact
#### Scenario: Dirty repository and staged work
- **WHEN** a confirmed multifile patch edits working files containing earlier manual changes
- **THEN** snapshots preserve the immediate preimage and the Git index remains unchanged
#### Scenario: Interrupted application
- **WHEN** application restarts with an applying record
- **THEN** all-before means no write, all-after means applied, mixed known states recover to before, and any unknown state is reported as conflict without overwriting
