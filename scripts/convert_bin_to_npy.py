from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert real IQ .bin files to device_*/.npy chunks")
    p.add_argument("--input-dir", required=True, help="Folder containing .bin files")
    p.add_argument("--output-dir", required=True, help="Output folder for device_*/.npy")
    p.add_argument(
        "--samples-per-chunk",
        type=int,
        default=4096,
        help="Complex IQ samples per output chunk",
    )
    p.add_argument(
        "--dtype",
        choices=["int16", "float32"],
        default="int16",
        help="Raw numeric type stored in .bin",
    )
    p.add_argument(
        "--max-chunks-per-device",
        type=int,
        default=400,
        help="Cap chunks per device for fast smoke tests",
    )
    return p.parse_args()


def extract_device_id(path: Path) -> int:
    m = re.search(r"device[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    digits = re.findall(r"\d+", path.stem)
    if digits:
        return int(digits[-1])
    raise ValueError(f"Cannot infer device id from filename: {path.name}")


def to_complex_iq(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 1:
        raw = raw.reshape(-1)
    if raw.size < 2:
        return np.zeros((0,), dtype=np.complex64)
    if raw.size % 2 == 1:
        raw = raw[:-1]
    iq = raw.reshape(-1, 2)
    comp = (iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32)).astype(np.complex64)
    return comp


def convert_file(
    file_path: Path,
    out_root: Path,
    dtype: str,
    samples_per_chunk: int,
    max_chunks_per_device: int,
) -> int:
    np_dtype = np.int16 if dtype == "int16" else np.float32
    raw = np.fromfile(file_path, dtype=np_dtype)
    iq = to_complex_iq(raw)

    device_id = extract_device_id(file_path)
    device_dir = out_root / f"device_{device_id}"
    device_dir.mkdir(parents=True, exist_ok=True)

    total = iq.shape[0]
    if total < samples_per_chunk:
        return 0

    n_chunks = min(total // samples_per_chunk, max_chunks_per_device)
    for i in range(n_chunks):
        s = i * samples_per_chunk
        e = s + samples_per_chunk
        chunk = iq[s:e]
        out_path = device_dir / f"sample_{i:06d}.npy"
        np.save(out_path, chunk)
    return n_chunks


def main() -> None:
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        raise SystemExit(f"Input directory not found: {in_dir}")

    bin_files = sorted(in_dir.rglob("*.bin"))
    if not bin_files:
        raise SystemExit(f"No .bin files found under: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    converted_files = 0
    for f in bin_files:
        try:
            n = convert_file(
                file_path=f,
                out_root=out_dir,
                dtype=args.dtype,
                samples_per_chunk=args.samples_per_chunk,
                max_chunks_per_device=args.max_chunks_per_device,
            )
            if n > 0:
                converted_files += 1
                total_chunks += n
                print(f"converted {f.name}: {n} chunks")
            else:
                print(f"skipped {f.name}: not enough samples")
        except Exception as ex:
            print(f"failed {f.name}: {ex}")

    print(
        f"done: converted_files={converted_files}, total_chunks={total_chunks}, output={out_dir}"
    )


if __name__ == "__main__":
    main()
