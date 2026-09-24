from __future__ import annotations

import argparse
import socket
import sys
import os
from pathlib import Path
from typing import Optional

# Tambahkan root project ke sys.path agar engine.config_loader bisa diimport
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from engine.config_loader import load_camera_config, CameraConfig


# ═══════════════════════════════════════════════════════════════════════════════
# ANSI Colors (Windows compat via colorama-free approach)
# ═══════════════════════════════════════════════════════════════════════════════

def _ok(s: str) -> str:
    return f"\033[92m{s}\033[0m"   # Hijau

def _fail(s: str) -> str:
    return f"\033[91m{s}\033[0m"  # Merah

def _warn(s: str) -> str:
    return f"\033[93m{s}\033[0m"  # Kuning


# ═══════════════════════════════════════════════════════════════════════════════
# TAHAP 1: Socket Check
# ═══════════════════════════════════════════════════════════════════════════════


def check_socket(host: str, port: int, timeout: float = 3.0) -> tuple[bool, str]:
    """
    Cek apakah TCP socket bisa dibuka ke host:port.

    Returns:
        (success: bool, message: str)
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, f"OPEN ✓"
    except socket.timeout:
        return False, f"TIMEOUT ✗ (tidak ada respons dalam {timeout:.0f}s)"
    except ConnectionRefusedError:
        return False, "REFUSED ✗ (port ditolak — kamera mungkin offline)"
    except OSError as exc:
        return False, f"ERROR ✗ ({exc})"


# ═══════════════════════════════════════════════════════════════════════════════
# TAHAP 2: RTSP DESCRIBE Handshake
# ═══════════════════════════════════════════════════════════════════════════════


def check_rtsp_describe(rtsp_url: str, host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """
    Kirim RTSP DESCRIBE request dan periksa response code.

    Args:
        rtsp_url: URL RTSP lengkap (sudah ter-resolve dengan credentials)
        host    : Hostname/IP kamera
        port    : Port RTSP (biasanya 554)
        timeout : Timeout koneksi dan read dalam detik

    Returns:
        (success: bool, message: str)
    """
    request = (
        f"DESCRIBE {rtsp_url} RTSP/1.0\r\n"
        f"CSeq: 1\r\n"
        f"User-Agent: FacilityManagementDiagnostic/1.0\r\n"
        f"\r\n"
    )

    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode("utf-8"))
            response = sock.recv(1024).decode("utf-8", errors="ignore")

        # Parse status line: "RTSP/1.0 200 OK"
        lines = response.splitlines()
        if not lines:
            return False, "NO RESPONSE ✗ (tidak ada respons dari kamera)"

        status_line = lines[0].strip()

        if "200" in status_line:
            return True, f"200 OK ✓ — Stream siap"
        elif "401" in status_line:
            return False, "401 Unauthorized ✗ — Periksa credentials (CAM_XX_USER / CAM_XX_PASS)"
        elif "403" in status_line:
            return False, "403 Forbidden ✗ — Akses ditolak oleh kamera"
        elif "404" in status_line:
            return False, "404 Not Found ✗ — Path RTSP tidak valid (periksa CAM_XX_PATH)"
        elif "RTSP" in status_line:
            return False, f"Respons tidak dikenal ✗ — {status_line}"
        else:
            return False, f"Respons non-RTSP ✗ — {status_line[:80]}"

    except socket.timeout:
        return False, f"READ TIMEOUT ✗ (timeout {timeout:.0f}s saat membaca respons)"
    except ConnectionRefusedError:
        return False, "CONNECTION REFUSED ✗"
    except OSError as exc:
        return False, f"SOCKET ERROR ✗ ({exc})"


# ═══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTIC RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _extract_host_port(cfg: CameraConfig) -> tuple[str, int]:
    """
    Ekstrak host dan port dari rtsp_url yang sudah ter-resolve.
    Fallback ke env vars CAM_XX_HOST / CAM_XX_PORT jika parse gagal.
    """
    # Coba parse dari URL
    try:
        # Format: rtsp://user:pass@host:port/path
        without_scheme = cfg.rtsp_url.replace("rtsp://", "")
        at_idx = without_scheme.rfind("@")
        host_part = without_scheme[at_idx + 1:].split("/")[0]
        if ":" in host_part:
            host, port_str = host_part.rsplit(":", 1)
            return host, int(port_str)
        return host_part, 554
    except Exception:
        # Fallback ke env var
        cam_prefix = cfg.camera_id.upper().replace("-", "_")
        host = os.environ.get(f"{cam_prefix}_HOST", "unknown")
        port = int(os.environ.get(f"{cam_prefix}_PORT", "554"))
        return host, port


def check_opencv_capture(rtsp_url: str, timeout_sec: float = 5.0) -> tuple[bool, str]:
    """
    Validasi stream RTSP dengan full digest authentication handshake via OpenCV.
    """
    import cv2
    try:
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_sec * 1000))
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                return True, f"AUTH & STREAM OK ✓ — Frame diterima ({w}x{h})"
            return True, "AUTH & STREAM OK ✓ — Stream terbuka"
        cap.release()
        return False, "GAGAL ✗ — Kredensial ditolak atau stream timeout"
    except Exception as exc:
        return False, f"ERROR ✗ ({exc})"


def diagnose_camera(cfg: CameraConfig, verbose: bool = False) -> bool:
    """
    Jalankan kedua tahap diagnostic untuk satu kamera.

    Returns:
        True jika semua tahap berhasil.
    """
    cam_id = cfg.camera_id
    host, port = _extract_host_port(cfg)

    print(f"\n{'─' * 60}")
    print(f"  Kamera : {cam_id} — {cfg.display_name}")
    print(f"  Host   : {host}:{port}")
    print(f"{'─' * 60}")

    # Tahap 1: Socket
    sock_ok, sock_msg = check_socket(host, port)
    print(f"  [{cam_id}] Socket {host}:{port:<6}  →  {_ok(sock_msg) if sock_ok else _fail(sock_msg)}")

    if not sock_ok:
        print(f"  {_warn('⚠  Socket gagal — melewati RTSP DESCRIBE.')}")
        return False

    # Tahap 2: RTSP DESCRIBE
    rtsp_ok, rtsp_msg = check_rtsp_describe(cfg.rtsp_url, host, port)

    # Jika kamera membutuhkan RTSP Digest Auth (401), validasi via OpenCV
    if not rtsp_ok and "401" in rtsp_msg:
        print(f"  [{cam_id}] RTSP Socket Handshake  →  {_ok('REACHABLE ✓')} (Digest Auth Challenge 401)")
        cv_ok, cv_msg = check_opencv_capture(cfg.rtsp_url)
        print(f"  [{cam_id}] RTSP Digest Auth       →  {_ok(cv_msg) if cv_ok else _fail(cv_msg)}")
        rtsp_ok = cv_ok
    else:
        print(f"  [{cam_id}] RTSP DESCRIBE          →  {_ok(rtsp_msg) if rtsp_ok else _fail(rtsp_msg)}")

    if verbose and rtsp_ok:
        print(f"  [INFO] AI canvas target: {cfg.resolution.ai_width}x{cfg.resolution.ai_height}")

    return rtsp_ok



def run_diagnostics(camera_ids: Optional[list[str]] = None, verbose: bool = False) -> int:
    """
    Jalankan diagnostic untuk semua atau kamera tertentu.

    Returns:
        Exit code: 0 = semua OK, 1 = ada kamera bermasalah
    """
    cameras_dir = PROJECT_ROOT / "cameras"

    # Temukan semua direktori kamera (bukan _template)
    cam_dirs = [
        d for d in cameras_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_")
        and (d / "config.json").exists()
    ]

    if not cam_dirs:
        print(_warn("Tidak ada kamera terkonfigurasi di cameras/"))
        return 1

    # Filter berdasarkan argumen --camera jika ada
    if camera_ids:
        cam_dirs = [d for d in cam_dirs if d.name in camera_ids]
        if not cam_dirs:
            print(_fail(f"Kamera tidak ditemukan: {camera_ids}"))
            return 1

    print(f"\n{'═' * 60}")
    print(f"  FACILITY MANAGEMENT — RTSP Diagnostic Tool")
    print(f"  Kamera ditemukan: {len(cam_dirs)}")
    print(f"{'═' * 60}")

    results: dict[str, bool] = {}
    for cam_dir in sorted(cam_dirs):
        config_path = cam_dir / "config.json"
        try:
            cfg = load_camera_config(config_path)
            results[cfg.camera_id] = diagnose_camera(cfg, verbose=verbose)
        except Exception as exc:
            print(f"\n  {_fail(f'Error loading config {cam_dir.name}: {exc}')}")
            results[cam_dir.name] = False

    # Summary
    print(f"\n{'═' * 60}")
    print("  SUMMARY")
    print(f"{'─' * 60}")
    all_ok = True
    for cam_id, ok in results.items():
        icon = _ok("PASS") if ok else _fail("FAIL")
        print(f"  {icon}  {cam_id}")
        if not ok:
            all_ok = False

    print(f"{'═' * 60}")
    if all_ok:
        print(_ok("  Semua kamera OK ✓"))
    else:
        print(_fail("  Satu atau lebih kamera bermasalah. Periksa log di atas."))
    print()

    return 0 if all_ok else 1


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Facility Management — RTSP Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--camera", "-c",
        nargs="+",
        metavar="CAM_ID",
        help="ID kamera yang akan diperiksa (default: semua kamera)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Tampilkan informasi tambahan",
    )
    args = parser.parse_args()

    # Aktifkan ANSI color di Windows terminal
    os.system("")

    exit_code = run_diagnostics(camera_ids=args.camera, verbose=args.verbose)
    sys.exit(exit_code)
