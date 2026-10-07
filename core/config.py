"""Конфигурация ff.ai: пути, переменные окружения и технические границы.

Предметной области здесь нет: домен (область применимости) целиком живёт в пакете
`domains/<id>/` и загружается `core/domains.py`. Этот модуль — только механика:
где лежит состояние пользователя, куда идти за моделью, каковы границы настроек ответа.

Все переменные окружения начинаются с `FFAI_` (кроме ключа и адреса облачного API,
унаследованных от донора), чтобы любой прогон — тест, демонстрация — можно было целиком
изолировать от реальных файлов пользователя.
"""

from __future__ import annotations

import hashlib
import os
from configparser import ConfigParser
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH)

ASSETS_DIR = BASE_DIR / "assets"
DOMAINS_DIR = BASE_DIR / "domains"


# --- Состояние приложения: вне целевого репозитория -------------------------


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def _xdg_base(env_name: str, fallback: str) -> Path:
    value = os.getenv(env_name)
    return Path(value) if value else _home() / fallback


def _env_path(name: str, default: Path) -> Path:
    value = os.getenv(name)
    return Path(value) if value else default


STATE_DIR = _env_path(
    "FFAI_STATE_DIR", _xdg_base("XDG_DATA_HOME", ".local/share") / "ff-ai"
)
CACHE_DIR = _env_path("FFAI_CACHE_DIR", _xdg_base("XDG_CACHE_HOME", ".cache") / "ff-ai")

HISTORY_FILE = _env_path("FFAI_HISTORY_FILE", STATE_DIR / "history.json")
MEMORY_FILE = _env_path("FFAI_MEMORY_FILE", STATE_DIR / "memory.json")
PROFILE_FILE = _env_path("FFAI_PROFILE_FILE", STATE_DIR / "profile.json")
TASK_FILE = _env_path("FFAI_TASK_FILE", STATE_DIR / "task.json")
SCHEDULE_FILE = _env_path("FFAI_SCHEDULE_FILE", STATE_DIR / "schedule.json")
TASKS_DIR = _env_path("FFAI_TASKS_DIR", STATE_DIR / "tasks")
EXPORTS_DIR = _env_path("FFAI_EXPORTS_DIR", STATE_DIR / "reports")


def repo_index_file(root: Path) -> Path:
    """Путь индекса для целевого репозитория: свой подкаталог кэша на каждый репозиторий."""
    override = os.getenv("FFAI_INDEX_FILE")
    if override:
        return Path(override)
    digest = hashlib.sha1(str(Path(root).resolve()).encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / digest / "index.sqlite3"


# --- Домен ------------------------------------------------------------------

# Идентификаторов домена в ядре нет: какой пакет активен и какой из них домен по умолчанию,
# решает `core/domains.py`, читая сами пакеты (`domains/<id>/domain.json`).
DOMAIN_ENV = os.getenv("FFAI_DOMAIN", "").strip()
REPO_ROOT_ENV = os.getenv("FFAI_REPO_ROOT", "").strip()


# --- Облачный и локальный API ----------------------------------------------

DEFAULT_API_URL = "https://opencode.ai/zen/v1/chat/completions"
API_URL = os.getenv("OPENCODE_API_URL", DEFAULT_API_URL)
DEFAULT_LOCAL_API_URL = "http://127.0.0.1:9999/v1/chat/completions"
LOCAL_API_URL = os.getenv("FFAI_LOCAL_API_URL", DEFAULT_LOCAL_API_URL)
DEFAULT_EMBEDDINGS_URL = "http://127.0.0.1:9999/v1/embeddings"

REQUEST_TIMEOUT = int(os.getenv("FFAI_REQUEST_TIMEOUT", "90"))
# Таймаут подключения к MCP-серверу: поиск по порталу отвечает секундами, но медленный
# сервер не должен держать интерфейс бесконечно.
MCP_TIMEOUT = float(os.getenv("FFAI_MCP_TIMEOUT", "60"))
# Имя сервера, которым окружение заменяет реестр целиком. Тесты и демонстрации подставляют
# сюда имя того сервера, который изображают: иначе домен не найдёт свой корпус документации.
MCP_OVERRIDE_NAME = os.getenv("FFAI_MCP_NAME", "из окружения")
MAX_RETRIES = 3
TRANSIENT_STATUSES = (502, 503, 504)

LLAMA_MODELS_FILE = BASE_DIR / "llama_server" / "models.ini"


def _local_presets(path: Path) -> ConfigParser:
    parser = ConfigParser()
    if path.exists():
        parser.read(path, encoding="utf-8")
    return parser


def local_models(path: Path = None) -> List[str]:
    """Имена пресетов локальных чат-моделей из models.ini (без embedding-пресетов)."""
    parser = _local_presets(path or LLAMA_MODELS_FILE)
    names = []
    for section in parser.sections():
        if parser.getboolean(section, "embedding", fallback=False):
            continue
        names.append(section)
    return names


def local_embedding_models(path: Path = None) -> List[str]:
    parser = _local_presets(path or LLAMA_MODELS_FILE)
    return [
        section
        for section in parser.sections()
        if parser.getboolean(section, "embedding", fallback=False)
    ]


LOCAL_MODELS = local_models()
LOCAL_EMBEDDING_MODELS = local_embedding_models()


def is_local_model(model: str) -> bool:
    return model in LOCAL_MODELS


def api_url_for_model(model: str) -> str:
    """Локальные пресеты идут на llama.cpp, всё остальное — в облачный сервис."""
    return LOCAL_API_URL if is_local_model(model) else API_URL


AVAILABLE_MODELS = [
    "deepseek-v4.1-flash",
    "deepseek-v4-pro",
    "glm-5.3-flash",
    "mimo-v2.5-free",
    "kimi-k3",
] + LOCAL_MODELS
DEFAULT_MODEL = AVAILABLE_MODELS[0]

# Цены входных/выходных токенов (доллары за 1 млн токенов) по действующему прайсу провайдера.
# Точность сравнительная, не бухгалтерская: оценка нужна, чтобы видеть порядок расхода.
MODEL_PRICING: Dict[str, Tuple[float, float]] = {
    "deepseek-v4.1-flash": (0.30, 1.20),
    "deepseek-v4-pro": (1.74, 3.48),
    "glm-5.3-flash": (0.15, 0.50),
    "mimo-v2.5-free": (0.0, 0.0),
    "kimi-k3": (3.00, 15.00),
}
MODEL_PRICING.update({model: (0.0, 0.0) for model in LOCAL_MODELS})


# --- Настройки ответа -------------------------------------------------------

TEMPERATURE = 0.7
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0

MIN_MAX_WORDS = 10
MAX_MAX_WORDS = 1000
DEFAULT_MAX_WORDS = 200

DEFAULT_LIST_LIMIT = 3
MIN_LIST_LIMIT = 1
MAX_LIST_LIMIT = 10

MIN_COMPRESS_AFTER = 5
MAX_COMPRESS_AFTER = 50
DEFAULT_COMPRESS_AFTER = int(os.getenv("FFAI_COMPRESS_AFTER", "10"))

MIN_MAX_SESSION_TOKENS = 5000
MAX_MAX_SESSION_TOKENS = 50000
DEFAULT_MAX_SESSION_TOKENS = int(os.getenv("FFAI_MAX_SESSION_TOKENS", "20000"))

# Потолок запроса к API: не длина ответа (её задаёт инструкция в промпте), а технический
# запас — reasoning-модели тратят сотни токенов до первого символа ответа.
WORDS_TO_TOKENS_RATIO = 4
TOKENS_OVERHEAD = 50
MIN_REQUEST_MAX_TOKENS = 2000


def max_tokens_for_words(max_words: int) -> int:
    return max(MIN_REQUEST_MAX_TOKENS, max_words * WORDS_TO_TOKENS_RATIO + TOKENS_OVERHEAD)


# --- Стратегии контекста ----------------------------------------------------

CONTEXT_STRATEGIES = ["summary", "sliding_window", "sticky_facts", "branching"]
DEFAULT_CONTEXT_STRATEGY = CONTEXT_STRATEGIES[0]

SUMMARY_MAX_WORDS = 150
FACTS_MAX_WORDS = 120
MAX_FACTS_KEYS = 20

MIN_PLAN_ITEMS = 3


# --- Прочее -----------------------------------------------------------------

# Корпус документации портала: сколько кандидатов берём из поиска, сколько фрагментов доставляем
# модели и как режем текст, чтобы документ не съел бюджет запроса.
DOCS_SEARCH_LIMIT = int(os.getenv("FFAI_DOCS_SEARCH_LIMIT", "5"))
DOCS_MAX_FRAGMENTS = int(os.getenv("FFAI_DOCS_MAX_FRAGMENTS", "3"))
DOCS_FRAGMENT_CHARS = int(os.getenv("FFAI_DOCS_FRAGMENT_CHARS", "1500"))
DOCS_SNIPPET_CHARS = int(os.getenv("FFAI_DOCS_SNIPPET_CHARS", "400"))
# Вторая ступень отбора: режим (оценивать кандидатов моделью или нет) и порог оценки.
DOCS_RETRIEVAL_MODE = os.getenv("FFAI_DOCS_RETRIEVAL_MODE", "enhanced")
DOCS_RELEVANCE_THRESHOLD = float(os.getenv("FFAI_DOCS_RELEVANCE_THRESHOLD", "0.6"))
DOCS_RERANK_MAX_WORDS = int(os.getenv("FFAI_DOCS_RERANK_MAX_WORDS", "2000"))

# Проверка ссылок: минимальная длина цитаты и число повторов до замены ответа.
DOCS_CITATION_MIN_CHARS = int(os.getenv("FFAI_DOCS_CITATION_MIN_CHARS", "20"))

# --- Корпус кода ---
# Поиск по коду локальный: векторы считаются из текста фрагментов, поэтому режим и порог влияют
# только на отбор, а не на то, ходит ли приложение в сеть.
CODE_RETRIEVAL_MODE = os.getenv("FFAI_CODE_RETRIEVAL_MODE", "enhanced")
CODE_RELEVANCE_THRESHOLD = float(os.getenv("FFAI_CODE_RELEVANCE_THRESHOLD", "0.6"))
CODE_CANDIDATES_BEFORE = int(os.getenv("FFAI_CODE_CANDIDATES_BEFORE", "20"))
CODE_FRAGMENTS_AFTER = int(os.getenv("FFAI_CODE_FRAGMENTS_AFTER", "3"))
# Сколько текста кандидата видит оценщик и сколько доставляется в запрос ответа.
CODE_SNIPPET_CHARS = int(os.getenv("FFAI_CODE_SNIPPET_CHARS", "1200"))
CODE_FRAGMENT_MAX_CHARS = int(os.getenv("FFAI_CODE_FRAGMENT_MAX_CHARS", "4000"))
CODE_RERANK_MAX_WORDS = int(os.getenv("FFAI_CODE_RERANK_MAX_WORDS", "2000"))
CODE_QUERY_MAX_WORDS = int(os.getenv("FFAI_CODE_QUERY_MAX_WORDS", "200"))
# Минимум дословной цитаты кода выше, чем у документации: короткая строка кода подтверждает
# что угодно и ничего не доказывает.
CODE_CITATION_MIN_CHARS = int(os.getenv("FFAI_CODE_CITATION_MIN_CHARS", "40"))

# --- Автовызов инструментов ---
# Автовызов стоит одного вспомогательного запроса на вопрос; выключение (для тестов, считающих
# запросы, и для работы без инструментов) не делает ни одного запроса выбора.
AUTO_TOOLS = os.getenv("FFAI_AUTO_TOOLS", "1").strip().lower() not in ("0", "false", "нет")
# Потолок ответа выбора: рассуждающие модели тратят сотни токенов до первого символа JSON.
TOOL_CHOICE_MAX_WORDS = int(os.getenv("FFAI_TOOL_CHOICE_MAX_WORDS", "2000"))
# Шагов в одной цепочке и раундов всего: пределы — данные, а не константы в коде флоу.
TOOL_CHAIN_MAX_STEPS = int(os.getenv("FFAI_TOOL_CHAIN_MAX_STEPS", "4"))
TOOL_FLOW_MAX_ROUNDS = int(os.getenv("FFAI_TOOL_FLOW_MAX_ROUNDS", "6"))
TOOL_FLOW_MAX_STEPS = int(os.getenv("FFAI_TOOL_FLOW_MAX_STEPS", "10"))
# Бюджеты контекста: результаты последнего раунда целиком, ранние — коротко, каждый шаг в запросе
# ответа — с ограничением.
TOOL_FLOW_CONTEXT_CHARS = int(os.getenv("FFAI_TOOL_FLOW_CONTEXT_CHARS", "24000"))
TOOL_FLOW_OLD_RESULT_CHARS = int(os.getenv("FFAI_TOOL_FLOW_OLD_RESULT_CHARS", "1500"))
TOOL_ANSWER_RESULT_CHARS = int(os.getenv("FFAI_TOOL_ANSWER_RESULT_CHARS", "6000"))

# --- Задача с проверкой (P6) ---
# Потолки этапов: у каждой операции свой предел длины, и он же задан в тексте запроса — модель
# получает то, что от неё ждут, а не «напиши раздел».
TASK_PLAN_MAX_WORDS = int(os.getenv("FFAI_TASK_PLAN_MAX_WORDS", "400"))
TASK_PATCH_MAX_WORDS = int(os.getenv("FFAI_TASK_PATCH_MAX_WORDS", "1500"))
TASK_VALIDATE_MAX_WORDS = int(os.getenv("FFAI_TASK_VALIDATE_MAX_WORDS", "500"))
TASK_GOAL_MAX_CHARS = int(os.getenv("FFAI_TASK_GOAL_MAX_CHARS", "400"))
TASK_RESULT_SLUG_MAX_CHARS = int(os.getenv("FFAI_TASK_RESULT_SLUG_MAX_CHARS", "40"))
# Первая версия плана ограничена, дальше предел живёт на задаче и растёт вместе с названной работой.
MAX_PLAN_ITEMS = int(os.getenv("FFAI_MAX_PLAN_ITEMS", "10"))
MAX_PLAN_ROUNDS = int(os.getenv("FFAI_MAX_PLAN_ROUNDS", "5"))
MAX_VALIDATION_ATTEMPTS = int(os.getenv("FFAI_MAX_VALIDATION_ATTEMPTS", "3"))
MAX_TRANSITION_LOG = int(os.getenv("FFAI_MAX_TRANSITION_LOG", "20"))
DOCS_CITATION_RETRIES = int(os.getenv("FFAI_DOCS_CITATION_RETRIES", "1"))

MAX_INPUT_LENGTH = 2000
ESTIMATED_CHARS_PER_TOKEN = 3  # приближение для клиентской оценки токенов (см. core/usage.py)

_api_key_runtime: Optional[str] = None


def get_api_key() -> Optional[str]:
    return _api_key_runtime or os.getenv("OPENCODE_API_KEY")


def set_api_key_runtime(api_key: str) -> None:
    global _api_key_runtime
    _api_key_runtime = api_key
