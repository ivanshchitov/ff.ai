# Переименование команд корпусов: `/code` → `/rag-code`, `/docs` → `/rag-docs`

## Why

Обе команды — это RAG по разным корпусам: `/code` ищет по исходникам репозитория, `/docs` — по
документации портала. Прежние имена этого не показывали: `/docs` читается как «документы проекта», а
не «документация портала», и по имени нельзя догадаться, что обе команды работают с поиском и
цитатами. Новые имена говорят прямо: это два RAG-корпуса.

## What Changes

- Команды корпуса кода: `/rag-code index [fixed|structural]`, `/rag-code status`, `/rag-code compare`,
  `/rag-code retrieval baseline|enhanced`, `/rag-code threshold <0|1>`, `/rag-code tune before= after=`,
  `/rag-code mode on|off`, `/rag-code trace`.
- Команды корпуса документации: `/rag-docs`, `/rag-docs mode on|off`, `/rag-docs version <версия>`,
  `/rag-docs retrieval baseline|enhanced`, `/rag-docs threshold <0|1>`, `/rag-docs trace`.
- Префиксы `rag-` у обоих корпусов: две команды видны рядом в панели `/commands` и в автодополнении.
- Сообщения, подсказки и тексты ошибок называют новые имена; ничего другого не меняется — поведение
  поиска, отбора и проверки ссылок прежнее.

## Impact

- Правки: `core/session.py` (диспетчер и тексты), `ui/commands_screen.py`, `ui/tui_app.py`,
  сообщения об индексе и векторах, тесты (юнит и сквозные), `README.md`, `AGENTS.md`,
  `docs/roadmap.md`.
- Спецификации: обновляются два требования (`code-document-index`, `code-retrieval`), где имена
  команд названы прямо; команды корпуса документации в спецификациях по именам не упоминались.
- Виды отчётов web-слоя (`/api/reports/code`, `/api/reports/docs`) — это имена корпусов, а не команд:
  остаются как есть.
