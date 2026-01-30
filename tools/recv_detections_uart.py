#!/usr/bin/env python3
"""UART YOLO 검출 결과 수신 → detections.bin 저장."""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Receive YOLO detections from UART")
    ap.add_argument("--port", default=None, help="Serial port (e.g. COM3, /dev/ttyUSB0)")
    ap.add_argument("--baud", type=int, default=115200, help="Baud rate")
    ap.add_argument("--out", default="data/output/detections_uart.bin", help="Output .bin path")
    args = ap.parse_args()

    try:
        import serial
    except ImportError:
        print("pip install pyserial", file=sys.stderr)
        return 1

    if not args.port:
        print("--port required (e.g. COM3 or /dev/ttyUSB0)", file=sys.stderr)
        return 1

    ser = serial.Serial(args.port, args.baud, timeout=2.0)
    try:
        # "YOLO\n"
        while True:
            line = ser.readline().decode("ascii", errors="ignore").strip()
            if line == "YOLO":
                break
            if line:
                print("skip:", line)

        # count (2 hex chars)
        line = ser.readline().decode("ascii", errors="ignore").strip()
        count = int(line, 16) if line else 0

        # 12*count bytes as hex (one line, no spaces)
        line = ser.readline().decode("ascii", errors="ignore").strip()
        if not line or len(line) < 12 * count * 2:
            print("short payload", file=sys.stderr)
            return 1
        raw = bytes.fromhex(line[: 12 * count * 2])
    finally:
        ser.close()

    # detections.bin: 1 byte count + 12*count bytes
    out_path = args.out
    with open(out_path, "wb") as f:
        f.write(bytes([count]))
        f.write(raw)
    print(f"Saved {count} detections to {out_path} ({1 + len(raw)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
