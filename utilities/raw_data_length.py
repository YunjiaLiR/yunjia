"""Report the duration of an interleaved multichannel binary recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def recording_duration(
    file_path: Path,
    num_channels: int,
    sampling_rate: float,
    dtype: str = "int16",
) -> float:
    """Return recording duration in seconds."""
    bytes_per_sample = np.dtype(dtype).itemsize
    file_size_bytes = file_path.stat().st_size

    denominator = num_channels * bytes_per_sample
    if file_size_bytes % denominator != 0:
        raise ValueError(
            "File size is not an exact multiple of channels × bytes/sample. "
            "Check --channels and --dtype."
        )

    samples_per_channel = file_size_bytes // denominator
    return samples_per_channel / sampling_rate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calculate the duration of a multichannel binary recording."
    )
    parser.add_argument("file", type=Path, help="Path to the binary recording.")
    parser.add_argument("--channels", type=int, default=128, help="Number of channels.")
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=30000,
        help="Sampling rate in Hz.",
    )
    parser.add_argument("--dtype", default="int16", help="Binary sample dtype.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(args.file)

    seconds = recording_duration(
        args.file,
        num_channels=args.channels,
        sampling_rate=args.sampling_rate,
        dtype=args.dtype,
    )

    print(f"File: {args.file}")
    print(f"Duration: {seconds:.2f} s ({seconds / 60:.2f} min)")


if __name__ == "__main__":
    main()
