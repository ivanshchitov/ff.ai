# chat-development-routing Specification

## Purpose
Route natural chat requests to read-only answers, clarification or the existing confirmed development pipeline.

## Requirements

### Requirement: Natural chat requests use the existing development pipeline
The application SHALL implement the following contract: Semantic classification SHALL distinguish answer, investigate, develop and clarify with strict JSON and counted usage. Errors, unknown kinds or missing goals SHALL clarify without launching a task. Explicit no-write requests SHALL override an inferred development intent. Ordinary questions and direct ask SHALL remain read-only. Development SHALL create one task and signal the existing runner; a busy or awaiting task SHALL be preserved. Clarification SHALL retain its original request and never approve an unseen patch. TUI and /api/ask SHALL share submit; answer fields SHALL remain compatible. An optional persisted request_id SHALL replay the same task without a new launch during execution, after completion and after restart; identical text with a different id SHALL be a new goal. Explicit /task SHALL bypass classification.
#### Scenario: Authorized operation
- **WHEN** a valid request meets the stated preconditions
- **THEN** the existing facade performs the operation and reports its result
#### Scenario: Invalid or unavailable precondition
- **WHEN** validation, confirmation or a required resource is unavailable
- **THEN** the application reports the reason and preserves user data

#### Scenario: Questions and investigation
- **WHEN** the user asks “Что делает клиент?”, “Исследуй ошибку, ничего не меняй” or “Предложи исправление без изменений”
- **THEN** submit uses the existing read-only answer flow and creates no development task
#### Scenario: Explicit development in either language
- **WHEN** the user asks “Исправь ошибку”, “Добавь тест”, “Обнови документацию” or “Fix the reconnect logic”
- **THEN** valid semantic development classification creates one persisted task and signals the existing runner
#### Scenario: Clarification and malformed classification
- **WHEN** the user says “Сделай лучше” without enough context or classification fails its strict schema
- **THEN** submit asks for clarification without changing the project; the subsequent reply includes the original request and does not approve a patch
#### Scenario: Busy and replay
- **WHEN** a development task is active or awaiting confirmation and another submission arrives
- **THEN** a matching persisted request_id returns its task without a new launch; another goal returns busy without replacing the task
