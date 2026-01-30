#!/usr/bin/env python3
"""UART 검출 결과 수신 → detections.bin 저장 → detections.txt(.jpg) 생성."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Receive YOLO detections from UART and write detections.txt (+ .jpg)"
    )
    ap.add_argument("--port", required=True, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "output",
        help="Output directory for detections.bin, detections.txt, detections.jpg",
    )
    ap.add_argument("--no-viz", action="store_true", help="Skip detections.jpg")
    ap.add_argument(
        "--img",
        type=Path,
        default=PROJECT_ROOT / "data" / "image" / "zidane.jpg",
        help="Input image for visualization",
    )
    args = ap.parse_args()

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_bin = args.out_dir / "detections.bin"

    tools = PROJECT_ROOT / "tools"
    recv_script = tools / "recv_detections_uart.py"
    decode_script = tools / "decode_detections.py"

    if not recv_script.exists() or not decode_script.exists():
        print("Missing tools/recv_detections_uart.py or tools/decode_detections.py", file=sys.stderr)
        return 1

    # 1) UART 수신 → detections.bin
    r = subprocess.run(
        [
            sys.executable,
            str(recv_script),
            "--port", args.port,
            "--baud", str(args.baud),
            "--out", str(out_bin),
        ],
        cwd=str(PROJECT_ROOT),
    )
    if r.returncode != 0:
        return r.returncode

    # 2) detections.bin → detections.txt (및 detections.jpg)
    cmd = [
        sys.executable,
        str(decode_script),
        "--c-bin", str(out_bin),
        "--out-dir", str(args.out_dir),
        "--img", str(args.img),
    ]
    if args.no_viz:
        cmd.append("--no-viz")
    r = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
