"""Visualize channel geometry exported by Kilosort."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

# ==========================================
# 1. SET YOUR FOLDER PATH
# ==========================================
folder_path = '20260303' # Change to your actual Kilosort output folder

# ==========================================
# 2. LOAD CHANNEL MAP & POSITIONS
# ==========================================
print("Loading probe geometry files...")
try:
    # positions usually has shape (N_channels, 2) -> [X, Y] in micrometers
    positions = np.load(os.path.join(folder_path, 'channel_positions.npy'))
    
    # channel_map has the original channel indices
    channel_map = np.load(os.path.join(folder_path, 'channel_map.npy')).flatten()
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Make sure you are pointing to the directory with Kilosort's .npy files.")
    exit()

# Try to load the shank assignments. If they don't exist, we will just use a default value,
# because plotting the X/Y coordinates will naturally reveal the shanks visually anyway!
shank_file = os.path.join(folder_path, 'channel_shanks.npy')
if os.path.exists(shank_file):
    shanks = np.load(shank_file).flatten()
    print(f"Successfully loaded shank assignments. Found {len(np.unique(shanks))} unique shank(s).")
else:
    print("No 'channel_shanks.npy' found. Grouping visually by X-coordinate instead.")
    # If no shank file exists, channels on different shanks usually have vastly different X coordinates.
    # We can fake the shank ID based on the unique X positions for the sake of coloring the plot.
    unique_x = np.unique(positions[:, 0])
    shanks = np.zeros(len(positions))
    for i, x_val in enumerate(unique_x):
        shanks[positions[:, 0] == x_val] = i

# ==========================================
# 3. PLOT THE PROBE GEOMETRY
# ==========================================
plt.figure(figsize=(8, 10))

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
