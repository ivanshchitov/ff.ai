## Why
Undo restores only the recorded agent change.

## What Changes
- The /changes list and show commands SHALL expose record ids, statuses and diffs. Undo SHALL display a reverse diff and require confirmation, restore exact before content/existence/modes without changing the Git index, and remove only empty directories created by that change. Later manual edits or later applied changes on intersecting paths SHALL block the whole undo. Repeated undo SHALL make no writes. Interrupted undo SHALL recover conservatively from snapshots. Related task patches SHALL be marked rolled back and their task paused until explicit continuation.

## Capabilities
### New Capabilities
- `project-change-undo`: Undo restores only the recorded agent change.
### Modified Capabilities

## Impact
Core, existing session facade, UI contracts, tests and documentation.
