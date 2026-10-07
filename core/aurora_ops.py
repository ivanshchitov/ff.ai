"""Конвейер операций над целевым проектом: цель SDK, сборка, подпись, установка, запуск.

Модуль знает про платформу ровно то, что объявил пакет домена: путь к инструменту, образец цели,
шаблоны команд, признаки подписи и запрещённые команды. Запуск процессов — через переданный
исполнитель, поэтому все ветки конвейера проверяются без SDK, устройства и настоящей сборки.

Необратимые шаги (подпись, установка, запуск) не выполняются без подтверждения, а состояние выбора
лежит рядом с проектом и не попадает в отслеживаемые файлы.
"""

from __future__ import annotations

import fnmatch
import json
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .domains import DomainOps

STATUS_OK = "ok"
STATUS_FAILED = "failed"
STATUS_UNAVAILABLE = "unavailable"
STATUS_REFUSED = "refused"

OUTPUT_TAIL = 800
VERSION_TIMEOUT = 60

# Плейсхолдеры шаблонов домена: только `{слово}` подставляется, всё остальное остаётся как есть.
_PLACEHOLDER = __import__("re").compile(r"\{(\w+)\}")


@dataclass(frozen=True)
class OpsReport:
    """Итог шага конвейера: чем закончился, почему, что вывел и сколько занял."""

    step: str
    status: str
    reason: str = ""
    output: str = ""
    seconds: float = 0.0
    path: str = ""

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAILED

    def line(self) -> str:
        mark = {
            STATUS_OK: "✅",
            STATUS_FAILED: "⛔",
            STATUS_UNAVAILABLE: "⏭",
            STATUS_REFUSED: "🚫",
        }.get(self.status, "•")
        parts = [f"{mark} {self.step}: {self.status}"]
        if self.reason:
            parts.append(self.reason)
        if self.path:
            parts.append(self.path)
        if self.seconds:
            parts.append(f"{self.seconds:.1f}с")
        return " | ".join(parts)


@dataclass(frozen=True)
class OpsState:
    """Сохранённый выбор: архитектура, цель и версия SDK, с которой цель выбиралась."""

    architecture: str = ""
    target: str = ""
    sdk_version: str = ""


@dataclass(frozen=True)
class OpsStatus:
    """Снимок состояния конвейера: инструмент, версия, доступные цели, выбор и оговорки."""

    tool: str
    sdk_version: str = ""
    targets: Tuple[str, ...] = ()
    chosen_architecture: str = ""
    chosen_target: str = ""
    docs_version: str = ""
    notes: Tuple[str, ...] = ()
    error: str = ""

    @property
    def available(self) -> bool:
        return not self.error

    def lines(self) -> Tuple[str, ...]:
        lines = [f"Инструмент: {self.tool}"]
        if self.error:
            lines.append(f"⏭ {self.error}")
            return tuple(lines)
        lines.append(f"Версия SDK: {self.sdk_version or 'неизвестна'}")
        if self.docs_version:
            lines.append(f"Версия документации домена: {self.docs_version}")
        if self.targets:
            lines.append(f"Цели: {', '.join(self.targets)}")
        else:
            lines.append("Целей в SDK не найдено.")
        if self.chosen_target:
            lines.append(f"Выбрано: {self.chosen_target} ({self.chosen_architecture})")
        else:
            lines.append("Цель не выбрана.")
        lines.extend(f"⚠ {note}" for note in self.notes)
        return tuple(lines)


def _tail(text: str, limit: int = OUTPUT_TAIL) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


class AuroraOps:
    """Операции конвейера: выбор цели, сборка, подпись, проверка, установка, запуск.

    `run(argv, cwd, timeout)` выполняет процесс и возвращает (код возврата, вывод, секунды);
    `confirm(step, description)` спрашивает разрешение на необратимый шаг. Оба приходят снаружи:
    в тестах это заглушки, в приложении — `subprocess` и интерфейс.
    """

    def __init__(
        self,
        root: Path,
        ops: DomainOps,
        run: Optional[Callable[[Sequence[str], Path, int], Tuple[int, str, float]]] = None,
        confirm: Optional[Callable[[str, str], bool]] = None,
        env: Optional[Dict[str, str]] = None,
        docs_version: str = "",
        app_id: str = "",
    ) -> None:
        self.root = Path(root)
        self.ops = ops
        self.docs_version = docs_version
        self.app_id = app_id
        self._run = run
        self._confirm = confirm
        self._env = env if env is not None else dict(os.environ)
        self._steps: List[OpsReport] = []
        self._verified: Optional[str] = None
        self._targets: Tuple[str, ...] = ()
        self._sdk_version: str = ""
        self.state = self._load_state()

    # --- состояние --------------------------------------------------------------------------

    @property
    def state_path(self) -> Path:
        return self.root / self.ops.state_dir / "ops.json"

    @property
    def build_dir(self) -> Path:
        arch = self.state.architecture or self.ops.default_architecture
        return self.root / f"build_{arch}"

    def _load_state(self) -> OpsState:
        path = self.state_path
        if not path.is_file():
            return OpsState()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return OpsState()
        if not isinstance(raw, dict):
            return OpsState()
        return OpsState(
            architecture=str(raw.get("architecture", "")),
            target=str(raw.get("target", "")),
            sdk_version=str(raw.get("sdk_version", "")),
        )

    def _save_state(self, state: OpsState) -> None:
        """Пишет состояние рядом с проектом и добавляет путь в локальные исключения git.

        Файлы проекта при этом не правятся: `.git/info/exclude` — локальный список git, а не
        отслеживаемый файл.
        """
        self.state = state
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_path.write_text(
                json.dumps(
                    {
                        "architecture": state.architecture,
                        "target": state.target,
                        "sdk_version": state.sdk_version,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            return
        self._ensure_excluded()

    def _ensure_excluded(self) -> None:
        exclude = self.root / ".git" / "info" / "exclude"
        if not exclude.parent.is_dir():
            return
        entry = f"{self.ops.state_dir}/"
        try:
            current = exclude.read_text(encoding="utf-8") if exclude.is_file() else ""
        except OSError:
            return
        if entry in current.splitlines():
            return
        try:
            with exclude.open("a", encoding="utf-8") as handle:
                if current and not current.endswith("\n"):
                    handle.write("\n")
                handle.write(entry + "\n")
        except OSError:
            return

    # --- инструмент и цели -------------------------------------------------------------------

    @property
    def tool(self) -> str:
        value = self._env.get(self.ops.tool_env, "").strip() or self.ops.tool_default
        return os.path.expanduser(value)

    @property
    def steps(self) -> Tuple[OpsReport, ...]:
        return tuple(self._steps)

    def _tool_available(self) -> str:
        path = Path(self.tool)
        if not path.is_file():
            return f"инструмент сборки не найден: {self.tool}"
        if not os.access(path, os.X_OK):
            return f"инструмент сборки не исполняем: {self.tool}"
        return ""

    def _guard(self, argv: Sequence[str]) -> str:
        """Запрещённые команды отсекаются до запуска процесса: строка сверяется префиксом.

        Сравнивается подкоманда инструмента, а не весь argv: путь к `sfdk` — часть машины
        разработчика, и сверять его с правилами домена бессмысленно.
        """
        tokens = list(argv)
        if tokens and tokens[0] == self.tool:
            tokens = tokens[1:]
        joined = " ".join(tokens)
        for forbidden in self.ops.forbidden_commands:
            if joined == forbidden or joined.startswith(forbidden + " "):
                return f"команда «{forbidden}» запрещена правилами домена: состав SDK менять нельзя"
        return ""

    def _exec(self, argv: Sequence[str], cwd: Path, timeout: int) -> OpsReport:
        """Выполняет команду через переданный исполнитель, соблюдая запреты."""
        refusal = self._guard(argv)
        if refusal:
            return OpsReport(step=argv[0] if argv else "", status=STATUS_REFUSED, reason=refusal)
        if self._run is None:
            return OpsReport(
                step=argv[0] if argv else "",
                status=STATUS_UNAVAILABLE,
                reason="исполнитель команд недоступен",
            )
        code, output, seconds = self._run(list(argv), cwd, timeout)
        status = STATUS_OK if code == 0 else STATUS_FAILED
        reason = "" if status == STATUS_OK else f"код возврата {code}"
        return OpsReport(step=argv[0], status=status, reason=reason, output=_tail(output), seconds=seconds)

    def _template(self, name: str, **values: str) -> List[str]:
        """Подставляет значения в шаблон домена.

        Подстановка своя, а не `str.format`: в командах встречаются фигурные скобки самого
        инструмента (`--qf %{SIGPGP:pgpsig}`), и формат строки принял бы их за свои поля.
        """
        template = self.ops.step(name)
        if not template:
            return []
        return [_PLACEHOLDER.sub(lambda match: values.get(match.group(1), match.group(0)), part)
                for part in template]

    def read_targets(self, refresh: bool = False) -> Tuple[str, ...]:
        """Список целей читается у самого инструмента: свой список устарел бы молча."""
        if self._targets and not refresh:
            return self._targets
        error = self._tool_available()
        if error:
            return ()
        report = self._exec([self.tool, "tools", "list"], self.root, VERSION_TIMEOUT)
        if not report.ok:
            return ()
        # Образец домена описывает цель конкретной архитектуры; для перечисления целей из него
        # берётся общая форма.
        generic = self.ops.target_pattern.replace("{arch}", "*")
        names: List[str] = []
        for line in report.output.splitlines():
            token = line.replace("│", " ").replace("├", " ").replace("└", " ").replace("─", " ")
            for part in token.split():
                if _matches(part, generic):
                    names.append(part)
        self._targets = tuple(dict.fromkeys(names))
        return self._targets

    def read_sdk_version(self) -> str:
        if self._sdk_version:
            return self._sdk_version
        error = self._tool_available()
        if error:
            return ""
        report = self._exec([self.tool, "--version"], self.root, VERSION_TIMEOUT)
        if not report.ok:
            return ""
        for line in report.output.splitlines():
            if line.startswith("SDK_RELEASE="):
                self._sdk_version = line.split("=", 1)[1].strip()
                break
        return self._sdk_version

    def status(self, refresh: bool = False) -> OpsStatus:
        """Снимок состояния: инструмент, версия, доступные цели, выбор и оговорки."""
        error = self._tool_available()
        if error:
            return OpsStatus(tool=self.tool, error=error, docs_version=self.docs_version)
        version = self.read_sdk_version()
        targets = self.read_targets(refresh=refresh)
        notes: List[str] = []
        if self.docs_version and version and version != self.docs_version:
            notes.append(
                f"версия SDK {version} не совпадает с версией документации домена "
                f"{self.docs_version}: платформенные факты бери из документации своей версии"
            )
        if not targets:
            notes.append("список целей пуст: проверь установку SDK")
        # Цель считается выбранной только после явного выбора: иначе состояние показывало бы
        # «выбрано» там, где ничего не сохранено и сборка не пойдёт.
        chosen_arch = self.state.architecture or self.ops.default_architecture
        chosen_target = self.state.target
        if not chosen_target:
            notes.append(
                f"цель не выбрана: домен предлагает архитектуру {chosen_arch} "
                f"({self.ops.architecture_note(chosen_arch)})"
            )
        return OpsStatus(
            tool=self.tool,
            sdk_version=version,
            targets=targets,
            chosen_architecture=chosen_arch,
            chosen_target=chosen_target,
            docs_version=self.docs_version,
            notes=tuple(notes),
        )

    def _target_for(self, arch: str, targets: Sequence[str] = None) -> str:
        candidates = targets if targets is not None else self.read_targets()
        pattern = self.ops.target_pattern.replace("{arch}", arch)
        for name in candidates:
            if _matches(name, pattern):
                return name
        return ""

    def select_target(self, arch: str) -> OpsReport:
        """Выбирает цель по архитектуре: цели, которой нет в SDK, не существует и для нас."""
        known = {name for name, _ in self.ops.architectures}
        if arch and arch not in known:
            return self._publish(
                OpsReport(
                    step="target",
                    status=STATUS_REFUSED,
                    reason=(
                        f"архитектура {arch} не объявлена доменом: "
                        f"доступны {', '.join(sorted(known))}"
                    ),
                )
            )
        chosen_arch = arch or self.ops.default_architecture
        targets = self.read_targets(refresh=True)
        if not targets:
            return self._publish(
                OpsReport(
                    step="target",
                    status=STATUS_UNAVAILABLE,
                    reason="список целей SDK не получен: выберите цель, когда инструмент ответит",
                )
            )
        target = self._target_for(chosen_arch, targets)
        if not target:
            return self._publish(
                OpsReport(
                    step="target",
                    status=STATUS_REFUSED,
                    reason=f"в SDK нет цели для {chosen_arch}: доступны {', '.join(targets)}",
                )
            )
        self._save_state(
            OpsState(architecture=chosen_arch, target=target, sdk_version=self.read_sdk_version())
        )
        return self._publish(OpsReport(step="target", status=STATUS_OK, path=target))

    # --- шаги конвейера ---------------------------------------------------------------------

    def _publish(self, report: OpsReport) -> OpsReport:
        self._steps.append(report)
        return report

    def _need_target(self) -> Optional[OpsReport]:
        if self.state.target:
            return None
        return OpsReport(
            step="target",
            status=STATUS_UNAVAILABLE,
            reason="цель не выбрана: сначала /ops target <архитектура>",
        )

    def _confirm_step(self, step: str, description: str) -> Optional[OpsReport]:
        if self._confirm is None:
            return OpsReport(
                step=step,
                status=STATUS_REFUSED,
                reason="шаг не подтверждён: подтверждение даёт интерфейс",
            )
        if not self._confirm(step, description):
            return OpsReport(step=step, status=STATUS_REFUSED, reason="пользователь отказался")
        return None

    def build(self) -> OpsReport:
        """Сборка в отдельном каталоге: корень проекта не должен получать артефактов."""
        missing = self._need_target()
        if missing:
            return self._publish(missing)
        error = self._tool_available()
        if error:
            return self._publish(OpsReport(step="build", status=STATUS_UNAVAILABLE, reason=error))
        build_dir = self.build_dir
        try:
            build_dir.mkdir(parents=True, exist_ok=True)
        except OSError as failure:
            return self._publish(
                OpsReport(step="build", status=STATUS_FAILED, reason=f"каталог сборки не создан: {failure}")
            )
        values = {
            "project": str(self.root),
            "build_dir": str(build_dir),
            "target": self.state.target,
            "arch": self.state.architecture,
        }
        init = self._template("build_init", **values)
        if init:
            report = self._exec([self.tool, *init], build_dir, VERSION_TIMEOUT)
            if not report.ok:
                return self._publish(replace(report, step="build-init"))
        report = self._exec([self.tool, *self._template("build", **values)], build_dir, 3600)
        if not report.ok:
            return self._publish(replace(report, step="build"))
        package = self.find_package()
        if not package:
            return self._publish(
                OpsReport(
                    step="build",
                    status=STATUS_FAILED,
                    reason="сборка прошла, но пакет не найден",
                    output=report.output,
                    seconds=report.seconds,
                )
            )
        return self._publish(
            OpsReport(
                step="build",
                status=STATUS_OK,
                output=report.output,
                seconds=report.seconds,
                path=str(package),
            )
        )

    def find_package(self) -> Optional[Path]:
        """Пакет ищется в каталоге сборки, а затем в проекте: первый по времени изменения."""
        candidates: List[Path] = []
        for base in (self.build_dir, self.root):
            if not base.is_dir():
                continue
            for pattern in self.ops.package_globs:
                candidates.extend(path for path in base.glob(pattern) if path.is_file())
        candidates = [path for path in dict.fromkeys(candidates)]
        if not candidates:
            return None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def sign(self) -> OpsReport:
        """Подпись пакета: фраза берётся из окружения, подтверждение — у пользователя."""
        key = self._env.get(self.ops.signature_key_env, "")
        if not key:
            return self._publish(
                OpsReport(
                    step="sign",
                    status=STATUS_UNAVAILABLE,
                    reason=(
                        f"кодовая фраза не задана: переменная {self.ops.signature_key_env} пуста, "
                        "интерактивный ввод в конвейере не используется"
                    ),
                )
            )
        package = self.find_package()
        if package is None:
            return self._publish(
                OpsReport(step="sign", status=STATUS_UNAVAILABLE, reason="пакет не найден: сначала сборка")
            )
        refused = self._confirm_step("sign", f"подписать {package.name}")
        if refused:
            return self._publish(refused)
        values = {"package": str(package), "key": key}
        report = self._exec([self.tool, *self._template("sign", **values)], self.root, 600)
        report = replace(report, step="sign", path=str(package) if report.ok else "")
        return self._publish(report)

    def verify(self) -> OpsReport:
        """Проверка подписи: решение по строкам вывода, а не по коду возврата."""
        package = self.find_package()
        if package is None:
            return self._publish(
                OpsReport(step="verify", status=STATUS_UNAVAILABLE, reason="пакет не найден")
            )
        values = {"package": str(package)}
        report = self._exec([self.tool, *self._template("verify", **values)], self.root, 600)
        if not report.ok:
            return self._publish(replace(report, step="verify"))
        lowered = report.output.casefold()
        unsigned = [marker for marker in self.ops.signature_unsigned if marker.casefold() in lowered]
        signed = [marker for marker in self.ops.signature_signed if marker.casefold() in lowered]
        if unsigned or not signed:
            return self._publish(
                OpsReport(
                    step="verify",
                    status=STATUS_FAILED,
                    reason="подпись не подтверждена строками вывода",
                    output=report.output,
                    seconds=report.seconds,
                )
            )
        self._verified = str(package)
        return self._publish(
            OpsReport(
                step="verify",
                status=STATUS_OK,
                output=report.output,
                seconds=report.seconds,
                path=str(package),
            )
        )

    def install(self) -> OpsReport:
        """Установка: неподписанный пакет на устройство не попадает."""
        package = self.find_package()
        if package is None:
            return self._publish(
                OpsReport(step="install", status=STATUS_UNAVAILABLE, reason="пакет не найден")
            )
        if self._verified != str(package):
            return self._publish(
                OpsReport(
                    step="install",
                    status=STATUS_REFUSED,
                    reason="подпись пакета не подтверждена: сначала /ops verify",
                    path=str(package),
                )
            )
        refused = self._confirm_step("install", f"установить {package.name} на устройство")
        if refused:
            return self._publish(refused)
        report = self._exec([self.tool, *self._template("deploy", package=str(package))], self.root, 900)
        return self._publish(replace(report, step="install", path=str(package) if report.ok else ""))

    def run_app(self) -> OpsReport:
        """Запуск приложения на устройстве: штатным способом платформы."""
        if not self.app_id:
            return self._publish(
                OpsReport(
                    step="run",
                    status=STATUS_UNAVAILABLE,
                    reason="неизвестен идентификатор приложения: домен его не объявил",
                )
            )
        refused = self._confirm_step("run", f"запустить {self.app_id} на устройстве")
        if refused:
            return self._publish(refused)
        report = self._exec(
            [self.tool, *self._template("run", app_id=self.app_id)], self.root, 300
        )
        return self._publish(replace(report, step="run"))

    def lines(self, status: Optional[OpsStatus] = None) -> Tuple[str, ...]:
        """Отчёт: состояние и журнал шагов из снимка, без повторного запуска процессов."""
        snapshot = status if status is not None else self.status()
        lines = list(snapshot.lines())
        if self._steps:
            lines.append("Шаги:")
            lines.extend(f"    {report.line()}" for report in self._steps)
        else:
            lines.append("Шагов не выполнялось.")
        return tuple(lines)


def _matches(name: str, pattern: str) -> bool:
    """Сопоставление с образцом домена: `*` — любая часть имени."""
    return fnmatch.fnmatch(name, pattern)
