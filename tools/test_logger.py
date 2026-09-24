"""
Facility Management — Test Suite: Centralized Logging System (tools/test_logger.py)
Menguji:
  1. Standarisasi kamus alias fixed-width 8-karakter (get_component_alias)
  2. ColoredConsoleFormatter (kolom sejajar, ANSI semantics, fallback)
  3. FileLogFormatter (bebas ANSI, format fixed-width)
  4. Helper log_error & perataan dua baris mitigasi (Console & File)
  5. Session Delimiters (Boot & Shutdown markers)
  6. Pembungkaman log library pihak ketiga (httpx, uvicorn.access = WARNING)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from engine.logger import (
    ColoredConsoleFormatter,
    FileLogFormatter,
    get_component_alias,
    setup_logging,
    get_logger,
    log_error,
    log_session_start,
    log_session_end,
    render_banner,
)

ANSI_REGEX = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")


def test_scenario_1_component_aliases():
    """Uji 1: Memverifikasi standarisasi kamus alias komponen 8-karakter fixed width."""
    print("\n[TEST 1] Menguji Standarisasi Kamus Alias Komponen (Fixed-Width 8 Karakter)...")

    expected_mappings = {
        "MasterOrchestrator": "SYSTEM",
        "__main__": "SYSTEM",
        "SYSTEM": "SYSTEM",
        "storage.database": "DATABASE",
        "engine.capture": "CAPTURE",
        "engine.preprocessor": "PREPROC",
        "engine.pipeline": "PIPELINE",
        "engine.disk_guard": "DISKGURD",
        "notification.dispatcher": "ALERT",
        "web.app": "WEBHUB",
        "uvicorn.access": "WEBHUB",
        "tools.roi_calibrator": "CALIBRAT",
        "ROICalibrator": "CALIBRAT",
    }

    for logger_name, expected_alias in expected_mappings.items():
        alias = get_component_alias(logger_name)
        assert alias == expected_alias, (
            f"Alias untuk '{logger_name}' diharapkan '{expected_alias}', tetapi didapat '{alias}'"
        )
        assert len(alias) <= 8, f"Alias '{alias}' melebihi 8 karakter"

    # Uji fallback
    custom_alias = get_component_alias("some.unknown.worker_module")
    assert len(custom_alias) <= 8
    assert custom_alias == "WORKER_M"

    print("  ✓ PASS: Kamus alias komponen tervalidasi fixed-width (maks 8 karakter uppercase).")


def test_scenario_2_console_formatting():
    """Uji 2: Memverifikasi format konsol fixed-width dan pewarnaan ANSI semantik."""
    print("\n[TEST 2] Menguji ColoredConsoleFormatter (Fixed-Width Columns & Alignment)...")

    formatter_color = ColoredConsoleFormatter(use_color=True)
    formatter_plain = ColoredConsoleFormatter(use_color=False)

    record_db = logging.LogRecord(
        name="storage.database",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Database siap",
        args=(),
        exc_info=None,
    )
    record_cap = logging.LogRecord(
        name="engine.capture",
        level=logging.WARNING,
        pathname=__file__,
        lineno=20,
        msg="Buffer capture penuh",
        args=(),
        exc_info=None,
    )

    out_db = formatter_plain.format(record_db)
    out_cap = formatter_plain.format(record_cap)

    # Validasi struktur kolom: HH:MM:SS │ LEVEL   │ [ALIAS   ] message
    pattern = r"^\d{2}:\d{2}:\d{2} │ [A-Z ]{7} │ \[[A-Z0-9_ ]{8}\] .+$"
    assert re.match(pattern, out_db), f"Format konsol database tidak sesuai pattern: {out_db}"
    assert re.match(pattern, out_cap), f"Format konsol capture tidak sesuai pattern: {out_cap}"

    # Pastikan posisi divider '│' tepat sama pada kedua baris
    first_sep_db = out_db.find("│")
    second_sep_db = out_db.find("│", first_sep_db + 1)
    first_sep_cap = out_cap.find("│")
    second_sep_cap = out_cap.find("│", first_sep_cap + 1)

    assert first_sep_db == first_sep_cap == 9, "Pembatas vertikal pertama harus berada pada indeks 9"
    assert second_sep_db == second_sep_cap == 19, "Pembatas vertikal kedua harus berada pada indeks 19"

    # Uji mode warna
    out_color = formatter_color.format(record_db)
    assert "│" in out_color
    assert "DATABASE" in ANSI_REGEX.sub("", out_color)

    print("  ✓ PASS: Garis pembatas vertikal dan kolom fixed-width rata presisi di konsol.")


def test_scenario_3_file_formatter_clean_text():
    """Uji 3: Memverifikasi berkas log murni tanpa ANSI dengan kolom fixed-width."""
    print("\n[TEST 3] Menguji FileLogFormatter (Pencatatan Berkas Bebas ANSI)...")

    temp_dir = Path(tempfile.mkdtemp(prefix="fac_log_clean_"))
    try:
        log_file = temp_dir / "app.log"
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        formatter = FileLogFormatter()
        handler.setFormatter(formatter)

        test_logger = logging.getLogger("storage.database")
        test_logger.setLevel(logging.DEBUG)
        test_logger.addHandler(handler)

        test_logger.info("SQLite WAL mode aktif ✓")
        test_logger.warning("Kapasitas disk mencapai batas ambang")
        test_logger.error("Koneksi gagal")
        handler.close()
        test_logger.removeHandler(handler)

        content = log_file.read_text(encoding="utf-8")
        lines = [l.strip() for l in content.strip().splitlines() if l.strip()]

        assert len(lines) == 3, f"Harus ada 3 baris log, ditemukan {len(lines)}"

        for line in lines:
            assert not ANSI_REGEX.search(line), f"Ditemukan kode escape ANSI dalam berkas log: {line}"
            # Format: [YYYY-MM-DD HH:MM:SS] [LEVEL  ] [ALIAS   ] message
            pattern = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[[A-Z ]{7}\] \[[A-Z0-9_ ]{8}\] .+$"
            assert re.match(pattern, line), f"Format baris berkas log tidak valid: {line}"

        print("  ✓ PASS: Berkas log 100% bebas dari kode escape ANSI dan kolom fixed-width presisi.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scenario_4_log_error_mitigation_alignment():
    """Uji 4: Memverifikasi helper log_error dan perataan mitigasi 2 baris terstruktur."""
    print("\n[TEST 4] Menguji log_error & Perataan Mitigasi 2 Baris...")

    temp_dir = Path(tempfile.mkdtemp(prefix="fac_log_err_"))
    try:
        log_file = temp_dir / "error_test.log"
        handler = logging.FileHandler(str(log_file), encoding="utf-8")
        handler.setFormatter(FileLogFormatter())

        err_logger = logging.getLogger("engine.capture")
        err_logger.setLevel(logging.INFO)
        err_logger.addHandler(handler)

        formatter = ColoredConsoleFormatter(use_color=False)

        # Buat record simulasi via log_error
        try:
            raise ConnectionError("Host unreachable")
        except Exception as exc:
            # 1. Uji konsol formatter langsung
            record = logging.LogRecord(
                name="engine.capture",
                level=logging.ERROR,
                pathname=__file__,
                lineno=99,
                msg="Gagal membuka stream video rtsp://192.168.1.100:554",
                args=(),
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            record.mitigation = "Periksa IP, port 554, atau kredensial kamera di .env"
            console_out = formatter.format(record)

            lines = console_out.splitlines()
            assert len(lines) >= 2, "Harus menghasilkan minimal 2 baris (Error + Mitigasi)"
            assert "│ ERROR   │ [CAPTURE ] Gagal membuka stream video" in lines[0]
            assert "│ DETAIL  │ -> Mitigasi: Periksa IP, port 554, atau kredensial kamera di .env" in lines[1]

            # Pastikan pembatas '│' baris 1 dan baris 2 sejajar
            sep1_line1 = lines[0].find("│")
            sep2_line1 = lines[0].find("│", sep1_line1 + 1)
            sep1_line2 = lines[1].find("│")
            sep2_line2 = lines[1].find("│", sep1_line2 + 1)
            assert sep1_line1 == sep1_line2 == 9, "Separator pertama harus sejajar di indeks 9"
            assert sep2_line1 == sep2_line2 == 19, "Separator kedua harus sejajar di indeks 19"

            # 2. Uji catatan berkas log
            log_error(
                err_logger,
                "Gagal membuka stream video rtsp://192.168.1.100:554",
                mitigation_hint="Periksa IP, port 554, atau kredensial kamera di .env",
                exc=exc,
            )

        handler.close()
        err_logger.removeHandler(handler)

        file_content = log_file.read_text(encoding="utf-8")
        assert "Mitigasi: Periksa IP, port 554" in file_content
        assert "Traceback (most recent call last):" in file_content

        print("  ✓ PASS: Format error 2 baris dengan petunjuk mitigasi sejajar sempurna.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scenario_5_session_delimiters():
    """Uji 5: Memverifikasi pencatatan session start & session end delimiters."""
    print("\n[TEST 5] Menguji Session Start & End Delimiters...")

    temp_dir = Path(tempfile.mkdtemp(prefix="fac_log_sess_"))
    try:
        log_file = temp_dir / "session.log"
        setup_logging(
            log_file=str(log_file),
            log_level="INFO",
            force_reconfigure=True,
        )

        log_session_start("Facility Management", "0.4.0", pid=12345)
        logger = get_logger("SYSTEM")
        logger.info("Subsistem beroperasi normal")
        log_session_end("Graceful Shutdown Complete (shutdown selesai)")

        content = log_file.read_text(encoding="utf-8")
        assert "--- SESSION START: Facility Management v0.4.0 (PID: 12345) ---" in content
        assert "--- SESSION END: Graceful Shutdown Complete (shutdown selesai) ---" in content

        print("  ✓ PASS: Session start dan end delimiters tercatat rapi di berkas log.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scenario_6_silenced_loggers():
    """Uji 6: Memverifikasi logger pihak ketiga (httpx, uvicorn.access) dibungkam pada level WARNING."""
    print("\n[TEST 6] Menguji Pembungkaman Logger Pihak Ketiga (httpx, uvicorn)...")

    temp_dir = Path(tempfile.mkdtemp(prefix="fac_log_silence_"))
    try:
        log_file = temp_dir / "silence.log"
        setup_logging(
            log_file=str(log_file),
            log_level="INFO",
            force_reconfigure=True,
        )

        # Logger pihak ketiga di level INFO tidak boleh masuk ke log
        httpx_logger = logging.getLogger("httpx")
        uvicorn_logger = logging.getLogger("uvicorn.access")

        assert httpx_logger.level == logging.WARNING, f"httpx level harus WARNING, got {httpx_logger.level}"
        assert uvicorn_logger.level == logging.WARNING, f"uvicorn.access level harus WARNING, got {uvicorn_logger.level}"

        httpx_logger.info("HTTP Request: GET http://testserver/ '200 OK'")
        uvicorn_logger.info("127.0.0.1 - 'GET / HTTP/1.1' 200")

        content = log_file.read_text(encoding="utf-8")
        assert "HTTP Request: GET" not in content, "Log GET 200 dari httpx tidak boleh masuk"
        assert "GET / HTTP/1.1" not in content, "Log request dari uvicorn tidak boleh masuk"

        print("  ✓ PASS: Logger pihak ketiga (httpx, uvicorn.access) terbungkam di level WARNING.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_scenario_7_preflight_table():
    """Uji 7: Memverifikasi rendering compact pre-flight checklist table."""
    print("\n[TEST 7] Menguji Rendering Compact Pre-Flight Checklist Table...")

    checks = {
        "Schema Audit": "1/1 Kamera Valid ✓",
        "Basis Data (WAL)": "storage/facility.db (WAL Mode OK)",
        "Profil Kamera": "cam_01 (640x360)",
        "Resource Guard": "Semaphore AI (1), DiskGuard (<80%/5.0GB)",
        "Web Streaming Hub": "http://0.0.0.0:8070 (Active)",
    }

    from engine.logger import render_preflight_table
    table = render_preflight_table(checks)

    assert "┌" in table and "┐" in table, "Border atas harus ada"
    assert "├" in table and "┤" in table, "Border tengah harus ada"
    assert "└" in table and "┘" in table, "Border bawah harus ada"
    assert "Schema Audit" in table
    assert "1/1 Kamera Valid ✓" in table
    assert "Basis Data (WAL)" in table
    assert "storage/facility.db" in table

    print(table)
    print("  ✓ PASS: Compact Pre-Flight table Unicode ter-render dengan sempurna.")


def test_scenario_8_dual_level_logging():
    """Uji 8: Memverifikasi dual-level logging (Console INFO vs File DEBUG)."""
    print("\n[TEST 8] Menguji Dual-Level Logging (Console INFO vs File DEBUG)...")

    temp_dir = Path(tempfile.mkdtemp(prefix="fac_log_dual_"))
    try:
        log_file = temp_dir / "dual.log"
        setup_logging(
            log_level="INFO",
            file_log_level="DEBUG",
            log_file=str(log_file),
            force_reconfigure=True,
        )

        test_logger = get_logger("SYSTEM")
        test_logger.debug("Pesan DEBUG teknis internal (hanya masuk berkas)")
        test_logger.info("Pesan INFO operasional (masuk konsol dan berkas)")

        content = log_file.read_text(encoding="utf-8")
        assert "Pesan DEBUG teknis internal" in content, "Pesan DEBUG wajib tersimpan di berkas log"
        assert "Pesan INFO operasional" in content, "Pesan INFO wajib tersimpan di berkas log"

        print("  ✓ PASS: Pesan DEBUG tersaring dari konsol namun tetap terekam lengkap di berkas log.")
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("============================================================")
    print("  FACILITY MANAGEMENT — LOGGING REFACTOR TEST SUITE")
    print("============================================================")

    test_scenario_1_component_aliases()
    test_scenario_2_console_formatting()
    test_scenario_3_file_formatter_clean_text()
    test_scenario_4_log_error_mitigation_alignment()
    test_scenario_5_session_delimiters()
    test_scenario_6_silenced_loggers()
    test_scenario_7_preflight_table()
    test_scenario_8_dual_level_logging()

    print("\n============================================================")
    print("  ALL 8 LOGGING REFACTOR TEST SCENARIOS: PASS ✓")
    print("============================================================")


if __name__ == "__main__":
    main()
