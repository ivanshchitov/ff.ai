"""Детерминированные проверки артефакта задачи по правилам домена.

Проверки идут перед модельным ревью и не зависят от него: модель, проверяющая свой же патч,
подтверждает свои же ошибки. Здесь ловится то, что формулируется правилом — запрещённые конструкции,
обязательные элементы описания пакета, границы путей, чистота применения и результат сборки.

Ни одно запрещённое слово не живёт в этом модуле: правила приходят из пакета домена.
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from . import patches
from .domains import DomainChecks

BUILD_OUTPUT_TAIL = 400


class BuildUnavailable(Exception):
    """Проверка сборки невозможна в этом окружении: это не «не прошло», а «не проверялось»."""


@dataclass(frozen=True)
class CheckIssue:
    """Нарушение: какой файл, какое правило и что именно нашлось."""

    path: str
    rule: str
    detail: str

    def line(self) -> str:
        return f"{self.path}: {self.rule} — {self.detail}"


@dataclass(frozen=True)
class CheckResult:
    """Итог проверок: нарушения, недоступные проверки и вывод сборки."""

    issues: Tuple[CheckIssue, ...] = ()
    skipped: Tuple[str, ...] = ()
    build_output: str = ""

    @property
    def ok(self) -> bool:
        return not self.issues

    def lines(self) -> Tuple[str, ...]:
        lines = [f"⛔ {issue.line()}" for issue in self.issues]
        lines.extend(f"⏭ {item}" for item in self.skipped)
        if self.build_output:
            lines.append(f"🔨 {self.build_output}")
        return tuple(lines)


def _matches_spec(path: str, globs: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in globs)


def check_forbidden(patch: patches.ParsedPatch, checks: DomainChecks) -> Tuple[CheckIssue, ...]:
    """Запрещённые конструкции в добавленных строках: удаление запрещённого — это исправление."""
    issues: List[CheckIssue] = []
    for item in patch.files:
        for line in item.added:
            for rule in checks.forbidden:
                if rule.matches(line):
                    issues.append(
                        CheckIssue(path=item.path, rule="запрещённая конструкция", detail=rule.reason)
                    )
                    break
    return tuple(issues)


def check_spec(patch: patches.ParsedPatch, checks: DomainChecks) -> Tuple[CheckIssue, ...]:
    """Описание пакета: у нового файла обязательные элементы должны быть, у правки — не пропасть.

    Проверяется ровно то, что видно в патче: для нового файла — его содержимое, для правки — что
    ни один обязательный элемент не удаляется. Полный разбор получившегося файла потребовал бы
    применения патча, а это уже не проверка, а изменение проекта.
    """
    if not checks.spec_required:
        return ()
    issues: List[CheckIssue] = []
    for item in patch.files:
        if not _matches_spec(item.path, checks.spec_globs):
            continue
        if item.is_new:
            missing = [
                field_name
                for field_name in checks.spec_required
                if not any(line.strip().startswith(field_name) for line in item.added)
            ]
            if missing:
                issues.append(
                    CheckIssue(
                        path=item.path,
                        rule="описание пакета",
                        detail="нет обязательных элементов: " + ", ".join(missing),
                    )
                )
            continue
        removed = [
            field_name
            for field_name in checks.spec_required
            if any(line.strip().startswith(field_name) for line in item.removed)
        ]
        if removed:
            issues.append(
                CheckIssue(
                    path=item.path,
                    rule="описание пакета",
                    detail="удалены обязательные элементы: " + ", ".join(removed),
                )
            )
    return tuple(issues)


def without_build(checks: DomainChecks) -> DomainChecks:
    """Копия правил без проверки сборки: она идёт на этапе проверки, а не на каждом патче.

    Сборка дороже всех остальных проверок вместе; запускать её после каждого применённого патча
    значило бы собирать проект шесть раз за задачу и всё равно судить по последней сборке.
    """
    return replace(checks, build_tool="", build_steps=(), build_description="")


def check_paths(patch: patches.ParsedPatch, root: Path) -> Tuple[CheckIssue, ...]:
    """Пути внутри целевого репозитория: всё, что выходит наружу, до применения недопустимо."""
    return tuple(
        CheckIssue(path=path, rule="путь вне репозитория", detail="изменение вне целевого проекта")
        for path in patches.outside_root(patch, root)
    )


def check_applies(root: Path, patch_text: str) -> Tuple[CheckIssue, ...]:
    """Чистота применения: патч, который не ложится на текущие файлы, применять нельзя."""
    reason = patches.check(root, patch_text)
    if not reason:
        return ()
    return (CheckIssue(path="(патч)", rule="применение", detail=reason),)


def check_build(
    checks: DomainChecks,
    build: Optional[Callable[[str], Tuple[bool, str]]],
) -> Tuple[Tuple[CheckIssue, ...], Tuple[str, ...], str]:
    """Сборка цели: команды домена выполняются переданным исполнителем.

    Исполнитель приходит снаружи (в приложении — инструмент сервера репозитория), поэтому белый
    список команд остаётся в одном месте. Отсутствие исполнителя или команды — не «пройдено», а
    названная недоступность: молча пропущенная проверка выглядит как успех.
    """
    if not checks.build_steps:
        return (), ("проверка сборки не объявлена доменом",), ""
    if build is None:
        return (), (f"проверка сборки недоступна: нет исполнителя команд (нужен {checks.build_tool})",), ""
    issues: List[CheckIssue] = []
    skipped: List[str] = []
    output: List[str] = []
    for step in checks.build_steps:
        try:
            ok, text = build(step)
        except BuildUnavailable as error:
            # Недоступность называется и останавливает проверку сборки: выдавать её за успех или
            # за провал одинаково неверно.
            skipped.append(f"проверка сборки недоступна: {error}")
            break
        except Exception as error:  # noqa: BLE001 - причина сбоя показывается текстом
            return (
                (CheckIssue(path="(сборка)", rule="сборка", detail=f"{step}: {error}"),),
                tuple(skipped),
                " ".join(output),
            )
        tail = (text or "").strip().replace("\n", " ")[-BUILD_OUTPUT_TAIL:]
        if tail:
            output.append(f"{step}: {tail}")
        if not ok:
            issues.append(
                CheckIssue(path="(сборка)", rule="сборка", detail=f"команда «{step}» завершилась ошибкой")
            )
            break
    return tuple(issues), tuple(skipped), " | ".join(output)


def check_patch(
    patch_text: str,
    root: Path,
    checks: DomainChecks,
    build: Optional[Callable[[str], Tuple[bool, str]]] = None,
    include_apply: bool = True,
) -> CheckResult:
    """Полный набор детерминированных проверок артефакта.

    Порядок — от дешёвого к дорогому: разбор, границы путей, запрещённые конструкции, описание
    пакета, применение и только затем сборка. Негодный на первых шагах патч не должен запускать
    сборку: она дороже всех остальных проверок вместе.
    """
    try:
        parsed = patches.parse(patch_text)
    except patches.PatchError as error:
        return CheckResult(issues=(CheckIssue(path="(патч)", rule="разбор", detail=str(error)),))

    issues: List[CheckIssue] = []
    issues.extend(check_paths(parsed, root))
    if issues:
        return CheckResult(issues=tuple(issues))
    issues.extend(check_forbidden(parsed, checks))
    issues.extend(check_spec(parsed, checks))
    if issues:
        return CheckResult(issues=tuple(issues))
    if include_apply:
        # Проверка «ложится ли патч» имеет смысл только до применения: у уже применённого
        # расхождения патч не «применяется» по определению, и это не нарушение.
        issues.extend(check_applies(root, patch_text))
        if issues:
            return CheckResult(issues=tuple(issues))
    build_issues, skipped, output = check_build(checks, build)
    return CheckResult(
        issues=tuple(build_issues), skipped=tuple(skipped), build_output=output
    )
