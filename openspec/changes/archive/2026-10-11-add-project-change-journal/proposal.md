## Why
Confirmed changes have a durable project journal.

## What Changes
- Text file creation, editing and deletion SHALL use one project-bound record with private before/after snapshots outside the project. Every patch header path SHALL stay inside the root and avoid symlinks and .git; binary and rename patches SHALL be rejected. Application SHALL require approval of the displayed diff, repeat fingerprints and domain checks after approval, leave the Git index unchanged and recover interrupted known states without overwriting unknown user edits. Standalone task execution SHALL use the same gate and deny writes without confirmation.

## Capabilities
### New Capabilities
- `project-change-journal`: Confirmed changes have a durable project journal.
### Modified Capabilities

## Impact
Core, existing session facade, UI contracts, tests and documentation.
