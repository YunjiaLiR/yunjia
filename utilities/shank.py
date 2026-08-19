import os
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. SET YOUR FOLDER PATH
# ==========================================
folder_path = '20260804' # Change to your actual Kilosort output folder

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
colors = plt.cm.get_cmap('tab10', len(unique_shanks)) # Automatically generate enough colors

# Scatter plot each shank's channels
for i, sh in enumerate(unique_shanks):
    mask = (shanks == sh)
    plt.scatter(
        positions[mask, 0], 
        positions[mask, 1], 
        color=colors(i), 
        label=f'Shank {int(sh)}', 
        s=40, 
        edgecolor='black', 
        linewidth=0.5
    )

# Add channel numbers as text labels next to the dots for ultimate debugging
for i, (x, y) in enumerate(positions):
    ch_id = channel_map[i]
    plt.text(x + 5, y, str(ch_id), fontsize=7, va='center')

plt.title("Probe Geometry & Shank Map (from Kilosort Data)")
plt.xlabel("X Position (µm)")
plt.ylabel("Y Position (µm)")

# Setting axis to 'equal' is CRITICAL so the probe doesn't look stretched/squished
plt.axis('equal') 
plt.legend(title="Detected Shanks", loc="upper right")
plt.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()

# Save the diagnostic plot
output_image = os.path.join(folder_path, "probe_geometry_diagnostic.png")
plt.savefig(output_image, dpi=300)
plt.show()

print(f"Saved diagnostic map to: {output_image}")