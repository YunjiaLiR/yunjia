"""Visualize channel geometry exported by Kilosort."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot Kilosort probe/shank geometry.")
    parser.add_argument(
        "folder",
        type=Path,
        help="Kilosort/Phy folder containing channel_positions.npy and channel_map.npy.",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional output path. Defaults to <folder>/probe_geometry_diagnostic.png.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    folder = args.folder

    positions_file = folder / "channel_positions.npy"
    map_file = folder / "channel_map.npy"

    if not positions_file.is_file() or not map_file.is_file():
        raise FileNotFoundError(
            "Expected channel_positions.npy and channel_map.npy in the supplied folder."
        )

    positions = np.load(positions_file)
    channel_map = np.load(map_file).flatten()

    if positions.shape[0] != len(channel_map):
        raise ValueError("channel_positions.npy and channel_map.npy have different lengths.")

    shank_file = folder / "channel_shanks.npy"
    if shank_file.is_file():
        shanks = np.load(shank_file).flatten()
        if len(shanks) != len(positions):
            raise ValueError("channel_shanks.npy does not match channel positions.")
    else:
        # Diagnostic fallback only: group by physical X coordinate.
        unique_x = np.unique(positions[:, 0])
        shanks = np.zeros(len(positions), dtype=int)
        for index, x_value in enumerate(unique_x):
            shanks[positions[:, 0] == x_value] = index

    fig, ax = plt.subplots(figsize=(8, 10))

    unique_shanks = np.unique(shanks)
    cmap = plt.colormaps.get_cmap("tab10")

    for index, shank_id in enumerate(unique_shanks):
        mask = shanks == shank_id
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            color=cmap(index % 10),
            label=f"Shank {int(shank_id)}",
            s=40,
            edgecolor="black",
            linewidth=0.5,
        )

    for index, (x_coord, y_coord) in enumerate(positions):
        ax.text(
            x_coord + 5,
            y_coord,
            str(channel_map[index]),
            fontsize=7,
            va="center",
        )

    ax.set_title("Probe geometry and channel map")
    ax.set_xlabel("X position (µm)")
    ax.set_ylabel("Y position (µm)")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(title="Detected shanks", loc="upper right")
    fig.tight_layout()

    output = args.save or (folder / "probe_geometry_diagnostic.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.show()

    print(f"Saved diagnostic map to: {output}")


if __name__ == "__main__":
    main()
