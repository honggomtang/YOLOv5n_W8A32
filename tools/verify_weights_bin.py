# -*- coding: utf-8 -*-
"""weights.bin 형식 검증 (C 로더와 동일 파싱)."""
from __future__ import annotations

import struct
import sys
from pathlib import Path


def verify(path: Path) -> bool:
    data = path.read_bytes()
    pos = 0
    end = len(data)
    if pos + 4 > end:
        print("Too short")
        return False
    num_tensors = struct.unpack_from("<I", data, pos)[0]
    pos += 4
    print(f"num_tensors: {num_tensors}")

    for i in range(num_tensors):
        if pos + 4 > end:
            print(f"Tensor {i}: truncated key_len")
            return False
        key_len = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if key_len > 1024 or pos + key_len > end:
            print(f"Tensor {i}: invalid key_len or truncated key")
            return False
        key = data[pos : pos + key_len].decode("utf-8", errors="replace")
        pos += key_len

        if pos + 4 > end:
            print(f"Tensor {i}: truncated ndim")
            return False
        ndim = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if ndim > 16 or pos + ndim * 4 > end:
            print(f"Tensor {i}: invalid ndim or truncated shape")
            return False
        shape = list(struct.unpack_from("<" + "I" * ndim, data, pos))
        pos += ndim * 4

        # 4-byte align (same as C loader)
        u = pos
        if u % 4 != 0:
            u = (u + 3) & ~3
            pos = u

        num_elems = 1
        for d in shape:
            num_elems *= d
        data_bytes = num_elems * 4
        if pos + data_bytes > end:
            print(f"Tensor {i} ({key}): truncated data (need {data_bytes})")
            return False
        pos += data_bytes
        if i < 3:
            print(f"  [{i}] {key!r} shape={tuple(shape)} data_bytes={data_bytes}")

    print(f"OK: {num_tensors} tensors, total read {pos} bytes, file {end} bytes")
    return True


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("assets/weights.bin")
    path = path.expanduser().resolve()
    if not path.exists():
        print(f"Not found: {path}")
        return 1
    return 0 if verify(path) else 1


if __name__ == "__main__":
    sys.exit(main())
