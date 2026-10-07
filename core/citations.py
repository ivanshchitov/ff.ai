"""Проверка ссылок и цитат: ответ обязан опираться на доставленные фрагменты.

Механизм общий для любого корпуса — и для документации портала (идентификатор фрагмента — путь
документа), и для кода целевого репозитория (идентификатор — путь с диапазоном строк). Модуль
не знает ни о MCP, ни о терминале: на вход текст ответа и список доставленных фрагментов.

Подтверждённой считается цитата, которая после нормализации регистра и пробелов не короче
минимума и встречается в тексте хотя бы одного доставленного фрагмента. Проверка намеренно
буквальная: она ловит выдуманный путь документа или имя API, но не оценивает смысл ответа —
смысловую сверку делает контрольный прогон. Приём тот же, что у инвариантов домена: инструкция
модели, проверка кодом, один повтор и замена ответа, если повтор тоже не подтверждён.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

from . import config

CITATIONS_ASSET = "docs_citations_prompt.md"

NO_SOURCE_MARKER = "источники не названы"
# Явный отказ от доставленных фрагментов: единственная разрешённая форма ответа без ссылки.
# Нужен потому, что домен разрешает отвечать на смежные вопросы (уровень 3) своими знаниями,
# а проверка иначе заменяла бы такой ответ: «документация к вопросу не относится» — это заявление
# модели, которое пользователь видит и может проверить.
OPT_OUT_MARKER = "Документация не относится к вопросу"
OPT_OUT_WINDOW = 200
NO_QUOTE_MARKER = "цитаты не подтверждены текстом доставленных фрагментов"

# Текст замены общий для корпусов: он говорит не о платформе и не о коде, а о том, что
# подтверждения нет. Доменные формулировки живут в пакете домена и в промптах.
DISCLAIMER_TEXT = (
    "Не знаю: в доставленных источниках нет фрагмента, который подтверждал бы ответ, а отвечать "
    "по памяти я не могу — без источника ответ нельзя проверить. Уточните, пожалуйста, вопрос: "
    "назовите точнее место или сущность, и я поищу снова."
)


class Citable(Protocol):
    """То, что можно процитировать: идентификатор, источник и текст доставленного фрагмента."""

    identifier: str
    source: str
    text: str


@dataclass(frozen=True)
class CitationsCheck:
    """Результат проверки: нарушения, был ли повтор, заменён ли ответ, был ли контекст вообще."""

    violations: Tuple[str, ...] = ()
    retried: bool = False
    replaced: bool = False
    final_violations: Tuple[str, ...] = ()
    no_context: bool = False
    opted_out: bool = False

    @property
    def confirmed(self) -> bool:
        return not self.violations and not self.replaced


def _normalize(text: str) -> str:
    """Нижний регистр и одиночные пробелы: цитата может отличаться переносами и регистром."""
    return " ".join(text.casefold().split())


def _has_quote(answer: str, fragments: Sequence[Citable], minimum: int = None) -> bool:
    """Есть ли в ответе окно доставленного фрагмента не короче минимума.

    Окно двигается от границы слова: так «...cache» в ответе не подтверждается окончанием
    «cache» внутри другого слова фрагмента.
    """
    normalized_answer = _normalize(answer)
    minimum = config.DOCS_CITATION_MIN_CHARS if minimum is None else minimum
    for fragment in fragments:
        text = _normalize(getattr(fragment, "text", "") or "")
        # Фрагмент короче минимума подтверждается цитатой целиком: иначе короткий файл нельзя было
        # бы подтвердить никогда, и ответ по нему гарантированно заменялся бы текстом «не знаю».
        minimum = min(minimum, len(text))
        if minimum == 0:
            continue
        for index in range(len(text) - minimum + 1):
            if index > 0 and text[index - 1] != " ":
                continue
            if text[index : index + minimum] in normalized_answer:
                return True
    return False


def _names_source(answer: str, fragments: Sequence[Citable]) -> bool:
    """Назван ли хотя бы один доставленный источник: путь документа или файл репозитория.

    Фрагмент кода опознаётся и по одному пути, без диапазона строк: «src/models.cpp» — это тот же
    источник, что «src/models.cpp:L10-L40», и требовать префикс `L` значило бы ловить формат записи,
    а не наличие ссылки. Замена, замена причины — нет: путь сверяется дословно.
    """
    normalized_answer = _normalize(answer)
    for fragment in fragments:
        identifier = getattr(fragment, "identifier", "") or ""
        candidates = [identifier, getattr(fragment, "source", "") or ""]
        if ":L" in identifier:
            candidates.append(identifier.split(":L", 1)[0])
        for candidate in candidates:
            candidate = _normalize(candidate)
            if candidate and candidate in normalized_answer:
                return True
    return False


def opted_out(answer: str) -> bool:
    """Ответ, который прямо говорит, что доставленные фрагменты к вопросу не относятся."""
    return OPT_OUT_MARKER.casefold() in answer[:OPT_OUT_WINDOW].casefold()


def check_answer(
    answer: str, fragments: Sequence[Citable], minimum: int = None
) -> CitationsCheck:
    """Проверяет ответ против доставленных фрагментов; пустой список — проверка не выполняется."""
    if not fragments:
        return CitationsCheck(no_context=True)
    if opted_out(answer):
        return CitationsCheck(opted_out=True)
    violations: List[str] = []
    if not _names_source(answer, fragments):
        violations.append(NO_SOURCE_MARKER)
    if not _has_quote(answer, fragments, minimum):
        violations.append(NO_QUOTE_MARKER)
    return CitationsCheck(violations=tuple(violations))


def check_groups(
    answer: str, groups: Sequence[Tuple[Sequence[Citable], int]]
) -> CitationsCheck:
    """Проверяет ответ против нескольких корпусов: источник и цитата — из любого из них.

    Корпуса различаются минимумом длины цитаты (документация — абзац, код — несколько строк),
    поэтому проверка идёт по каждому отдельно, а подтверждение любого из них достаточно:
    смешанный вопрос отвечается и по коду, и по документации, и требовать оба сразу значило бы
    запретить честный ответ по одному источнику.
    """
    delivered = tuple(
        (fragments, minimum) for fragments, minimum in groups if fragments
    )
    if not delivered:
        return CitationsCheck(no_context=True)
    if opted_out(answer):
        return CitationsCheck(opted_out=True)
    violations: List[str] = []
    for fragments, minimum in delivered:
        check = check_answer(answer, fragments, minimum)
        if check.confirmed:
            return CitationsCheck()
        violations.extend(check.violations)
    return CitationsCheck(violations=tuple(dict.fromkeys(violations)))


@lru_cache(maxsize=None)
def citations_message() -> str:
    """Инструкция цитат из ассета: она одинакова для всех корпусов, поэтому кэшируется."""
    path = config.ASSETS_DIR / CITATIONS_ASSET
    return path.read_text(encoding="utf-8").strip()


def context_message(fragments: Sequence[Citable], note: str = "") -> Optional[str]:
    """Системное сообщение с доставленными фрагментами: заголовок, текст, пометка об обрезке.

    Пустой список сообщения не даёт вовсе — тогда форма запроса остаётся прежней.
    """
    if not fragments:
        return None
    lines: List[str] = ["Ниже — фрагменты, найденные по запросу пользователя."]
    if note:
        lines.append(note)
    for number, fragment in enumerate(fragments, start=1):
        header = [f"[{number}]", getattr(fragment, "identifier", "")]
        section = getattr(fragment, "section", "")
        if section:
            header.append(f"раздел: {section}")
        version = getattr(fragment, "version", "")
        if version:
            header.append(f"версия: {version}")
        source = getattr(fragment, "source", "")
        if source:
            header.append(source)
        lines.append(" | ".join(part for part in header if part))
        lines.append(getattr(fragment, "text", "") or "")
        if getattr(fragment, "truncated", False):
            lines.append("(фрагмент обрезан)")
    return "\n".join(lines)


def retry_prompt(violations: Iterable[str]) -> str:
    """Требование повтора: что именно нарушено и что нужно сделать."""
    listed = "; ".join(violations)
    return (
        f"Ответ не подтверждён ({listed}). Перепиши ответ так, чтобы он называл путь доставленного "
        "документа и содержал дословную цитату из его текста — символ в символ, без пересказа. "
        "Если ни один доставленный фрагмент не отвечает на вопрос, скажи, что не знаешь, "
        "и попроси уточнить. Если вопрос решается без документации (общий вопрос о языке или "
        f"инструменте), начни ответ словами «{OPT_OUT_MARKER}» и отвечай как обычно."
    )


def disclaimer_text() -> str:
    return DISCLAIMER_TEXT
