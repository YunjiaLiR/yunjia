"""Plot a selected time interval from one channel of a raw binary recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect a raw electrophysiology channel.")
    parser.add_argument("file", type=Path, help="Path to the binary recording.")
    parser.add_argument("--channels", type=int, default=128)
    parser.add_argument("--sampling-rate", type=float, default=30000)
    parser.add_argument("--channel", type=int, default=0)
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds.")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds.")
    parser.add_argument("--bit-to-uv", type=float, default=0.25)
    parser.add_argument("--dtype", default="int16")
    parser.add_argument("--save", type=Path, default=None, help="Optional output image.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if not args.file.is_file():
        raise FileNotFoundError(args.file)
    if not 0 <= args.channel < args.channels:
        raise ValueError("--channel must be within the configured channel count.")

    dtype = np.dtype(args.dtype)
    samples_per_channel = int(args.duration * args.sampling_rate)
    values_to_read = samples_per_channel * args.channels

    start_sample = int(args.start * args.sampling_rate)
    offset_bytes = start_sample * args.channels * dtype.itemsize

    data = np.fromfile(
        args.file,
        dtype=dtype,
        count=values_to_read,
        offset=offset_bytes,
    )

    if data.size == 0:
        raise ValueError("No samples were read. Check the requested start time.")

    complete_rows = data.size // args.channels
    data = data[: complete_rows * args.channels].reshape(-1, args.channels)
    signal_uv = data[:, args.channel].astype(np.float32) * args.bit_to_uv

    time_s = args.start + np.arange(signal_uv.size) / args.sampling_rate

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time_s, signal_uv, linewidth=0.8)
    ax.set_title(f"Raw signal — channel {args.channel}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (µV)")
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=300)

    plt.show()


if __name__ == "__main__":
    main()
