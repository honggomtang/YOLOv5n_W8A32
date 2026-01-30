# -*- coding: utf-8 -*-
"""weights.bin 4바이트 정렬 패딩 추가 (RISC-V misalign 방지)."""
from __future__ import annotations

import argparse
import struct
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Add 4-byte alignment padding to weights.bin")
    ap.add_argument("--in", dest="in_path", default="assets/weights.bin", help="Input weights.bin")
    ap.add_argument("--out", default="assets/weights_aligned.bin", help="Output path")
    args = ap.parse_args()

    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    if not in_path.exists():
        print(f"Error: {in_path} not found")
        return 1

    data = in_path.read_bytes()
    pos = 0
    end = len(data)

    if pos + 4 > end:
        print("Error: file too short")
        return 1
    num_tensors = struct.unpack_from("<I", data, pos)[0]
    pos += 4

    out_chunks = [struct.pack("<I", num_tensors)]

    for _ in range(num_tensors):
        if pos + 4 > end:
            print("Error: truncated at tensor header")
            return 1
        key_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key_len > 1024 or pos + key_len > end:
            print("Error: invalid key_len or truncated key")
            return 1
        out_chunks.append(data[pos - 4 : pos + key_len])
        pos += key_len

        if pos + 4 > end:
            print("Error: truncated at ndim")
            return 1
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if ndim > 16 or pos + ndim * 4 > end:
            print("Error: invalid ndim or truncated shape")
            return 1
        out_chunks.append(data[pos - 4 : pos + ndim * 4])
        pos += ndim * 4

        # 현재 위치(헤더 끝) 기준으로 패딩 계산
        header_end = sum(len(c) for c in out_chunks)
        pad = (4 - (header_end % 4)) % 4
        if pad:
            out_chunks.append(b"\x00" * pad)

        num_elems = 1
        for j in range(ndim):
            num_elems *= struct.unpack_from("<I", data, pos - ndim * 4 + j * 4)[0]
        data_bytes = num_elems * 4
        if pos + data_bytes > end:
            print("Error: truncated tensor data")
            return 1
        out_chunks.append(data[pos : pos + data_bytes])
        pos += data_bytes

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"".join(out_chunks))
    print(f"Wrote {num_tensors} tensors to {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
