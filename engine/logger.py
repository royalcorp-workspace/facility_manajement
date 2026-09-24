"""
Facility Management — Centralized Logging Engine (engine/logger.py)
Menyediakan ColoredConsoleFormatter berbasis ANSI VT100 (Windows & Linux),
FileLogFormatter murni bebas ANSI, RotatingFileHandler terukur (5MB, 3 cadangan),
pemetaan alias fixed-width 8 karakter, session delimiters, serta helper log_error.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional

# ── Status Inisialisasi Singleton ─────────────────────────────────────────────
_logging_initialized: bool = False
_vt100_enabled: bool = False

# ── ANSI Escape Codes ─────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GRAY = "\033[90m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
BOLD_RED = "\033[1;31m"
BG_RED_BOLD = "\033[1;41;37m"
WHITE = "\033[37m"

LEVEL_COLORS: dict[int, str] = {
    logging.DEBUG: GRAY,
    logging.INFO: CYAN,
    logging.WARNING: YELLOW,
    logging.ERROR: BOLD_RED,
    logging.CRITICAL: BG_RED_BOLD,
}

# ── Kamus Standarisasi Alias Komponen (Fixed-Width 8 Karakter) ────────────────
COMPONENT_ALIASES: dict[str, str] = {
    "MasterOrchestrator": "SYSTEM",
    "__main__": "SYSTEM",
    "SYSTEM": "SYSTEM",
    "storage.database": "DATABASE",
    "engine.capture": "CAPTURE",
    "engine.preprocessor": "PREPROC",
    "engine.pipeline": "PIPELINE",
    "engine.disk_guard": "DISKGURD",
    "tools.roi_calibrator": "CALIBRAT",
    "ROICalibrator": "CALIBRAT",
    "SpatialRules": "SPATIAL",
    "engine.spatial_rules": "SPATIAL",
    "CROSSING": "CROSSING",
    "WALKWAY": "WALKWAY",
    "CONGEST": "CONGEST",
}


def get_component_alias(name: str) -> str:
    """Mengembalikan alias komponen dengan panjang tepat 8 karakter (uppercase)."""
    if name in COMPONENT_ALIASES:
        return COMPONENT_ALIASES[name]
    if name.startswith("notification.") or name == "notification":
        return "ALERT"
    if name.startswith("web.") or name == "web" or name.startswith("uvicorn"):
        return "WEBHUB"
    if name.startswith("storage."):
        return "DATABASE"
    if name.startswith("engine.capture"):
        return "CAPTURE"
    if name.startswith("engine.preprocessor"):
        return "PREPROC"
    if name.startswith("engine.pipeline"):
        return "PIPELINE"
    if name.startswith("engine.disk_guard"):
        return "DISKGURD"

    # Fallback: ambil segmen terakhir, uppercase, maks 8 char
    parts = name.split(".")
    short = parts[-1] if parts else name
    return short[:8].upper()


def enable_windows_vt100() -> bool:
    """Mengaktifkan ANSI Escape Processing (VT100) pada konsol Windows."""
    global _vt100_enabled
    if _vt100_enabled:
        return True

    if sys.platform == "win32":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # STD_OUTPUT_HANDLE = -11
            h_out = kernel32.GetStdHandle(-11)
            if h_out and h_out != -1:
                mode = ctypes.c_uint32()
                if kernel32.GetConsoleMode(h_out, ctypes.byref(mode)):
                    # ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                    kernel32.SetConsoleMode(h_out, mode.value | 0x0004)
                    _vt100_enabled = True
                    return True
        except Exception:
            pass
    else:
        _vt100_enabled = True
        return True
    return False


class ColoredConsoleFormatter(logging.Formatter):
    """
    Formatter konsol dengan pewarnaan semantik ANSI dan kolom fixed-width.
    Format: HH:MM:SS │ LEVEL   │ [ALIAS   ] message
            ↳ Baris kedua:         │ DETAIL  │ -> Mitigasi: ...
    """

    def __init__(self, use_color: bool = True) -> None:
        super().__init__()
        self.use_color = use_color and enable_windows_vt100()

    def format(self, record: logging.LogRecord) -> str:
        # Cek apakah stream dialihkan atau tidak mendukung warna
        use_color = self.use_color
        if not sys.stdout.isatty() and not os.environ.get("FORCE_COLOR"):
            use_color = False

        record_time = time.strftime("%H:%M:%S", time.localtime(record.created))
        levelname = f"{record.levelname:<7}"
        alias = f"{get_component_alias(record.name):<8}"

        if use_color:
            color = LEVEL_COLORS.get(record.levelno, WHITE)
            sep = f"{GRAY}│{RESET}"
            level_str = f"{color}{levelname}{RESET}"
            alias_str = f"{GRAY}[{RESET}{WHITE}{alias}{RESET}{GRAY}]{RESET}"
        else:
            sep = "│"
            level_str = levelname
            alias_str = f"[{alias}]"

        msg = record.getMessage()
        header = f"{record_time} {sep} {level_str} {sep} {alias_str} {msg}"

        # 1. Format mitigasi khusus jika disertakan via log_error / extra={"mitigation": ...}
        if hasattr(record, "mitigation") and record.mitigation:
            detail_tag = f"{CYAN}DETAIL {RESET}" if use_color else "DETAIL "
            header += f"\n         {sep} {detail_tag} {sep} -> Mitigasi: {record.mitigation}"

        # 2. Format exception info jika ada
        if record.exc_info:
            exc_type, exc_val, _ = record.exc_info
            exc_name = exc_type.__name__ if exc_type else "Error"
            if not (hasattr(record, "mitigation") and record.mitigation):
                detail_tag = f"{CYAN}DETAIL {RESET}" if use_color else "DETAIL "
                header += f"\n         {sep} {detail_tag} {sep} [REASON] {exc_name}: {exc_val}"
            trace_tag = f"{GRAY}TRACE  {RESET}" if use_color else "TRACE  "
            header += f"\n         {sep} {trace_tag} {sep} Full trace tercatat di berkas log"

        return header


class FileLogFormatter(logging.Formatter):
    """
    Formatter berkas murni tanpa kode ANSI dengan kolom fixed-width.
    Format: [YYYY-MM-DD HH:MM:SS] [%(levelname)-7s] [%(alias)-8s] %(message)s
    """

    def __init__(self) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        record_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.created))
        levelname = f"{record.levelname:<7}"
        alias = f"{get_component_alias(record.name):<8}"
        msg = record.getMessage()

        line = f"[{record_time}] [{levelname}] [{alias}] {msg}"

        if hasattr(record, "mitigation") and record.mitigation:
            line += f"\n                         ↳ Mitigasi: {record.mitigation}"

        if record.exc_info:
            if not record.exc_text:
                record.exc_text = self.formatException(record.exc_info)
            if record.exc_text:
                line += f"\n{record.exc_text}"

        return line


def setup_logging(
    log_level: str = "INFO",
    file_log_level: str = "DEBUG",
    log_file: str = "logs/app.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
    console_color: bool = True,
    force_reconfigure: bool = False,
) -> None:
    """
    Inisialisasi sistem logging terpusat Dual-Level:
    - Console Handler (stdout): Level INFO (bersih, event-driven)
    - RotatingFileHandler: Level DEBUG (telemetri forensik detail)
    - Root Logger: Level min(console, file)
    """
    global _logging_initialized
    if _logging_initialized and not force_reconfigure:
        return

    # Buat direktori logs jika belum ada
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    console_numeric = getattr(logging, log_level.upper(), logging.INFO)
    file_numeric = getattr(logging, file_log_level.upper(), logging.DEBUG)

    # Root logger diset ke level terendah agar pesan DEBUG mengalir ke file
    root_logger.setLevel(min(console_numeric, file_numeric))

    # Bersihkan handler sebelumnya jika ada
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # 1. Console Handler (Fixed-Width Colored, level INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_numeric)
    console_handler.setFormatter(ColoredConsoleFormatter(use_color=console_color))
    root_logger.addHandler(console_handler)

    # 2. Rotating File Handler (No ANSI, Fixed-Width, 5MB, level DEBUG)
    file_handler = RotatingFileHandler(
        str(log_path),
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(file_numeric)
    file_handler.setFormatter(FileLogFormatter())
    root_logger.addHandler(file_handler)

    # 3. Redam logger pihak ketiga yang bising
    for noisy in ("httpx", "httpcore", "uvicorn.access", "uvicorn.error", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _logging_initialized = True


def get_logger(name: str) -> logging.Logger:
    """Mengambil instance logger subsistem. Menginisialisasi default jika belum aktif."""
    if not _logging_initialized:
        env_level = os.environ.get("LOG_LEVEL", "INFO")
        env_file = os.environ.get("LOG_FILE", "logs/app.log")
        setup_logging(log_level=env_level, log_file=env_file)
    return logging.getLogger(name)


def log_error(
    component: str | logging.Logger,
    error_msg: str,
    mitigation_hint: Optional[str] = None,
    exc: Optional[Exception] = None,
) -> None:
    """
    Mencatat error dengan format terstruktur dua baris pada konsol (Error + Mitigasi)
    serta menyimpan traceback lengkap ke berkas log.
    """
    logger = component if isinstance(component, logging.Logger) else get_logger(component)
    extra = {"mitigation": mitigation_hint} if mitigation_hint else {}
    logger.error(error_msg, exc_info=exc, extra=extra)


# Alias untuk kompatibilitas backward
log_exception = log_error


def log_session_start(
    app_name: str = "Facility Management",
    version: str = "0.4.0",
    pid: Optional[int] = None,
) -> None:
    """Mencatat penanda sesi baru saat sistem boot."""
    current_pid = pid or os.getpid()
    delimiter = f"--- SESSION START: {app_name} v{version} (PID: {current_pid}) ---"
    logger = get_logger("SYSTEM")
    logger.info(delimiter)


def log_session_end(
    status: str = "Graceful Shutdown Complete (shutdown selesai)",
) -> None:
    """Mencatat penanda akhir sesi saat sistem berhenti."""
    delimiter = f"--- SESSION END: {status} ---"
    logger = get_logger("SYSTEM")
    logger.info(delimiter)


def render_banner(
    title: str,
    version: str,
    details: dict[str, str],
    width: int = 68,
) -> str:
    """
    Menghasilkan tampilan banner startup berbentuk compact card Unicode.
    """
    inner_width = width - 4
    top = f"┌{'─' * (width - 2)}┐"
    bottom = f"└{'─' * (width - 2)}┘"

    header_text = f"  {title} v{version}"
    header_line = f"│ {header_text:<{inner_width}} │"

    detail_parts = [f"{k}: {v}" for k, v in details.items()]
    detail_text = f"  {' │ '.join(detail_parts)}"
    detail_line = f"│ {detail_text:<{inner_width}} │"

    return f"\n{top}\n{header_line}\n{detail_line}\n{bottom}"


def render_preflight_table(
    checks: dict[str, str],
    col1_title: str = "KOMPONEN PRE-FLIGHT",
    col2_title: str = "STATUS / DETAIL KESIAPAN",
    width: int = 70,
) -> str:
    """
    Menghasilkan tabel checklist pre-flight berbingkai Unicode yang rapi dan elegan.
    """
    col1_width = 24
    col2_width = width - col1_width - 3

    top = f"┌{'─' * col1_width}┬{'─' * col2_width}┐"
    mid = f"├{'─' * col1_width}┼{'─' * col2_width}┤"
    bot = f"└{'─' * col1_width}┴{'─' * col2_width}┘"

    header = f"│ {col1_title:<{col1_width - 2}} │ {col2_title:<{col2_width - 2}} │"
    lines = [f"\n{top}", header, mid]

    for k, v in checks.items():
        val_str = str(v)
        if len(val_str) > col2_width - 2:
            val_str = val_str[:col2_width - 5] + "..."
        lines.append(f"│ {k:<{col1_width - 2}} │ {val_str:<{col2_width - 2}} │")

    lines.append(bot)
    return "\n".join(lines)
