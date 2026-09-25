"""
tools/check_multi_camera.py
===========================
Skrip diagnostik sekuensial multi-kamera untuk menguji:
1. TCP Socket Port 554 Reachability
2. RTSP DESCRIBE Protocol Handshake
3. OpenCV VideoCapture Decode Test (1 frame)
"""

from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env", override=False)

from engine.config_loader import load_camera_config, CameraConfig


def _ok(s: str) -> str:
    return f"\033[92m{s}\033[0m"

def _fail(s: str) -> str:
    return f"\033[91m{s}\033[0m"

def _warn(s: str) -> str:
    return f"\033[93m{s}\033[0m"

def _bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def check_tcp_socket(host: str, port: int = 554, timeout: float = 3.0) -> Tuple[bool, str, float]:
    """Cek koneksi TCP socket ke host:port dengan pengukuran latency RTT."""
    t0 = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            rtt_ms = (time.time() - t0) * 1000.0
            return True, f"OPEN ({rtt_ms:.1f}ms)", rtt_ms
    except socket.timeout:
        return False, f"TIMEOUT ({timeout:.0f}s)", 0.0
    except ConnectionRefusedError:
        return False, "REFUSED (Port tertutup)", 0.0
    except OSError as exc:
        return False, f"ERROR ({exc})", 0.0


def check_rtsp_describe(rtsp_url: str, host: str, port: int = 554, timeout: float = 4.0) -> Tuple[bool, str]:
    """Kirim RTSP DESCRIBE request untuk memverifikasi handshake protokol."""
    request = (
        f"DESCRIBE {rtsp_url} RTSP/1.0\r\n"
        f"CSeq: 1\r\n"
        f"User-Agent: MultiCameraDiagnostic/1.0\r\n"
        f"\r\n"
    )
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            sock.settimeout(timeout)
            sock.sendall(request.encode("utf-8"))
            resp = sock.recv(1024).decode("utf-8", errors="ignore")

        lines = resp.splitlines()
        if not lines:
            return False, "NO RESPONSE"
        status_line = lines[0].strip()

        if "200" in status_line:
            return True, f"200 OK"
        elif "401" in status_line:
            return True, f"401 Unauthorized (Digest Challenge - Reachable)"
        elif "403" in status_line:
            return False, f"403 Forbidden"
        elif "404" in status_line:
            return False, f"404 Stream Not Found"
        return False, status_line[:40]
    except Exception as exc:
        return False, f"FAILED ({exc})"


def check_opencv_decode(rtsp_url: str, timeout_sec: float = 4.0) -> Tuple[bool, str]:
    """Uji decode 1 frame aktual via OpenCV VideoCapture (FFMPEG)."""
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
                return True, f"DECODED ({w}x{h})"
            return True, "STREAM OPENED"
        cap.release()
        return False, "CANNOT OPEN"
    except Exception as exc:
        return False, f"ERROR ({exc})"


def parse_host_port_from_url(rtsp_url: str) -> Tuple[str, int]:
    try:
        without_scheme = rtsp_url.replace("rtsp://", "")
        at_idx = without_scheme.rfind("@")
        host_part = without_scheme[at_idx + 1:].split("/")[0]
        if ":" in host_part:
            h, p = host_part.rsplit(":", 1)
            return h, int(p)
        return host_part, 554
    except Exception:
        return "127.0.0.1", 554


def main():
    cameras_dir = PROJECT_ROOT / "cameras"
    cam_dirs = sorted([
        d for d in cameras_dir.iterdir()
        if d.is_dir() and not d.name.startswith("_") and (d / "config.json").exists()
    ])

    print("\n" + "=" * 75)
    print(_bold("  FACILITY MANAGEMENT -- Multi-Camera Network & RTSP Diagnostic"))
    print(f"  Target Kamera Terkonfigurasi: {len(cam_dirs)}")
    print("=" * 75)

    summary: List[Dict[str, str]] = []

    for cam_dir in cam_dirs:
        cfg_file = cam_dir / "config.json"
        try:
            cfg = load_camera_config(cfg_file)
        except Exception as exc:
            print(f"\n[{cam_dir.name}] Gagal memuat config: {exc}")
            summary.append({"cam_id": cam_dir.name, "host": "-", "socket": "ERROR", "rtsp": "ERROR", "decode": "ERROR"})
            continue

        host, port = parse_host_port_from_url(cfg.rtsp_url)
        classes_str = ", ".join(cfg.enabled_classes) if cfg.enabled_classes else "all (default)"

        print(f"\n--> Memeriksa {_bold(cfg.camera_id)} ({cfg.display_name})")
        print(f"    Host: {host}:{port} | Filter: [{classes_str}]")

        # 1. Socket Check
        sock_ok, sock_msg, rtt = check_tcp_socket(host, port, timeout=3.0)
        sock_fmt = _ok(sock_msg) if sock_ok else _fail(sock_msg)
        print(f"    [1/3] TCP Socket 554 : {sock_fmt}")

        if not sock_ok:
            print(f"    {_warn('Host tidak dapat dijangkau dari jaringan ini. Melewati RTSP decode.')}")
            summary.append({"cam_id": cfg.camera_id, "host": host, "socket": "FAIL", "rtsp": "SKIP", "decode": "SKIP"})
            continue

        # 2. RTSP DESCRIBE
        rtsp_ok, rtsp_msg = check_rtsp_describe(cfg.rtsp_url, host, port, timeout=4.0)
        rtsp_fmt = _ok(rtsp_msg) if rtsp_ok else _fail(rtsp_msg)
        print(f"    [2/3] RTSP Handshake : {rtsp_fmt}")

        # 3. OpenCV Decode
        cv_ok, cv_msg = check_opencv_decode(cfg.rtsp_url, timeout_sec=4.0)
        cv_fmt = _ok(cv_msg) if cv_ok else _fail(cv_msg)
        print(f"    [3/3] Video Decode   : {cv_fmt}")

        summary.append({
            "cam_id": cfg.camera_id,
            "host": host,
            "socket": "PASS" if sock_ok else "FAIL",
            "rtsp": "PASS" if rtsp_ok else "FAIL",
            "decode": "PASS" if cv_ok else "FAIL",
        })

    # Summary Table
    print("\n" + "=" * 75)
    print(_bold("  RINGKASAN DIAGNOSTIK MULTI-KAMERA"))
    print("-" * 75)
    print(f"  {'CAM ID':<10} │ {'HOST IP':<18} │ {'TCP 554':<10} │ {'RTSP DESC':<12} │ {'DECODE':<10}")
    print("-" * 75)
    for s in summary:
        s_sock = _ok("✓ OPEN") if s["socket"] == "PASS" else _fail("✗ DOWN")
        s_rtsp = _ok("✓ OK") if s["rtsp"] == "PASS" else (_warn("- SKIP") if s["rtsp"] == "SKIP" else _fail("✗ FAIL"))
        s_dec = _ok("✓ OK") if s["decode"] == "PASS" else (_warn("- SKIP") if s["decode"] == "SKIP" else _fail("✗ FAIL"))
        print(f"  {s['cam_id']:<10} │ {s['host']:<18} │ {s_sock:<19} │ {s_rtsp:<21} │ {s_dec}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
