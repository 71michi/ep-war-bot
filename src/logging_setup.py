import logging
import contextvars
from logging.handlers import RotatingFileHandler

from .config import env_str, env_int


# A per-message trace id that we inject into every log record.
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


class TraceIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id_var.get("-")  # type: ignore[attr-defined]
        return True


def set_trace_id(trace_id: str) -> contextvars.Token:
    return _trace_id_var.set(trace_id)


def reset_trace_id(token: contextvars.Token) -> None:
    try:
        _trace_id_var.reset(token)
    except Exception:
        pass


def get_trace_id() -> str:
    return _trace_id_var.get("-")


def setup_logging() -> None:
    """Configure console + (optional) log.txt with a trace id.

    IMPORTANT: We keep third-party debug logs OFF by default.
    - Console level controlled by LOG_LEVEL (default INFO)
    - File level controlled by LOG_FILE_LEVEL (default DEBUG)
    - File path controlled by LOG_FILE (default log.txt)

    The file handler is attached ONLY to the 'warbot' logger hierarchy, so the
    log stays readable (no base64 image payload dumps from HTTP/OpenAI libs).
    """
    root = logging.getLogger()
    if getattr(root, "_warbot_logging_configured", False):
        return

    console_level = env_str("LOG_LEVEL", "INFO").upper()
    file_level = env_str("LOG_FILE_LEVEL", "DEBUG").upper()
    log_file = env_str("LOG_FILE", "log.txt")

    max_bytes = env_int("LOG_MAX_BYTES", 5_000_000)
    backups = env_int("LOG_BACKUPS", 2)

    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(trace_id)s] %(name)s: %(message)s")
    flt = TraceIdFilter()

    # Root: console only (avoid flooding with third-party DEBUG logs)
    root.setLevel(getattr(logging, console_level, logging.INFO))

    sh = logging.StreamHandler()
    sh.setLevel(getattr(logging, console_level, logging.INFO))
    sh.setFormatter(fmt)
    sh.addFilter(flt)
    root.addHandler(sh)

    # warbot: detailed file logs
    wb = logging.getLogger("warbot")
    wb.setLevel(logging.DEBUG)
    wb.propagate = True  # INFO/WARN/ERROR still go to console via root

    try:
        fh = RotatingFileHandler(
            log_file,
            maxBytes=max(0, int(max_bytes)),
            backupCount=max(0, int(backups)),
            encoding="utf-8",
        )
        fh.setLevel(getattr(logging, file_level, logging.DEBUG))
        fh.setFormatter(fmt)
        fh.addFilter(flt)
        wb.addHandler(fh)
    except Exception:
        # If file logging fails (read-only FS etc.), keep console logging.
        pass

    # Silence noisy libraries (we log OpenAI calls ourselves at the warbot level).
    for noisy in (
        "openai",
        "openai._base_client",
        "httpx",
        "httpcore",
        "PIL",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    root._warbot_logging_configured = True  # type: ignore[attr-defined]
