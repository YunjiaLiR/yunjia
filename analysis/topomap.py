import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
binary_file = '20260303_data.bin'
folder_path = '20260303'
fs = 30000          
n_channels = 128    
dtype = 'int16'     

print("Loading metadata, spike files, and channel geometry...")
spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')

# Load the physical X and Y coordinates of the 128 channels
# channel_positions shape is usually (128, 2) -> [X_coord, Y_coord]
channel_positions = np.load(os.path.join(folder_path, 'channel_positions.npy'))

# --- NEW POST-CURATION FILTERING ---
metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)

keep_mask = (
    (metrics["isi_violations_ratio"] < 0.5) & 
    (metrics["amplitude_cutoff"] < 0.1) & 
    (metrics["presence_ratio"] > 0.8) & (metrics["amplitude_median"].abs() > 20)
)

# Keep this as a DataFrame so the rest of your script doesn't break!
good_units_df = metrics[keep_mask]
print(f"Identified {len(good_units_df)} 'good' single neurons from objective post-curation.")

# Determine the exact recording duration
file_bytes = os.path.getsize(binary_file)
bytes_per_data_point = np.dtype(dtype).itemsize 
total_samples = file_bytes // (n_channels * bytes_per_data_point)
duration_seconds = total_samples / fs

# ==========================================
# 2. DYNAMIC HARDWARE REMAPPING USING KILOSORT MAP
# ==========================================
print("Loading Kilosort hardware channel map...")
# channel_map.npy tells us exactly which 0-based hardware channel belongs to each row of channel_positions
channel_map = np.load(os.path.join(folder_path, 'channel_map.npy')).flatten()

full_n_channels = 128
full_channel_positions = np.empty((full_n_channels, 2))
full_channel_positions[:] = np.nan  # Fill with NaN to safely track omitted channels
full_channel_firing_rates = np.zeros(full_n_channels)

print(f"Mapping {len(channel_map)} Kilosort-active channels onto the 128-channel layout...")

# Dynamically populate physical positions using Kilosort's native map
for file_idx, hw_ch in enumerate(channel_map):
    if hw_ch < full_n_channels:
        full_channel_positions[hw_ch] = channel_positions[file_idx]

# Accumulate the firing rates using absolute hardware channel indices from your tsv
for index, row in good_units_df.iterrows():
    cluster_id = int(index) 
    
    # Extract the channel row from cluster_info
    ch_row = cluster_info[cluster_info['cluster_id'] == cluster_id]
    
    # --- CRITICAL FIX: Extract the ch_idx value ---
    if not ch_row.empty:
        # Pulls 'ch' or 'channel' depending on what your tsv uses
        ch_idx = int(ch_row['ch'].values[0]) if 'ch' in ch_row.columns else int(ch_row['channel'].values[0])
    else:
        print(f"Warning: Cluster {cluster_id} not found in cluster_info.tsv. Skipping.")
        continue
    # ----------------------------------------------
    
    n_spikes = np.sum(spike_clusters == cluster_id)
    firing_rate_hz = n_spikes / duration_seconds
    
    if ch_idx < full_n_channels:
        full_channel_firing_rates[ch_idx] += firing_rate_hz
    else:
        print(f"Warning: Cluster {cluster_id} claims to be on channel {ch_idx}, which exceeds {full_n_channels}!")

# Filter down strictly to the channels that Kilosort actively used 
used_mask = ~np.isnan(full_channel_positions[:, 0])
x_coords = full_channel_positions[used_mask, 0]
y_coords = full_channel_positions[used_mask, 1]
final_firing_rates = full_channel_firing_rates[used_mask]

# ==========================================
# 3. PLOT SPATIAL TOPOMAP 
# ==========================================
fig, ax = plt.subplots(figsize=(7, 6))

x_spread = max(x_coords) - min(x_coords)
y_spread = max(y_coords) - min(y_coords)

if x_spread > 50 and y_spread > 50:
    # 2D SMOOTH INTERPOLATED TOPOMAP (For Grids/Utah Arrays)
    print("Generating a smooth 2D interpolated topomap...")
    grid_x, grid_y = np.mgrid[min(x_coords):max(x_coords):200j, min(y_coords):max(y_coords):200j]
    
    clean_points = np.column_stack((x_coords, y_coords))
    grid_z = griddata(clean_points, final_firing_rates, (grid_x, grid_y), method='cubic')
    
    im = ax.imshow(grid_z.T, extent=(min(x_coords), max(x_coords), min(y_coords), max(y_coords)),
                   origin='lower', cmap='magma', aspect='auto')
    ax.scatter(x_coords, y_coords, c='white', s=15, alpha=0.3, edgecolors='none', zorder=3)
else:
    # DISCRETE GEOMETRY SCATTER MAP (For High-Density Linear Probes)
    print("Generating a high-density discrete channel map...")
    im = ax.scatter(x_coords, y_coords, c=final_firing_rates, cmap='magma', 
                    s=120, edgecolor='#CBD5E0', linewidth=0.5, zorder=2)

# Styling and Colorbar Layout
ax.set_title("Spatial Distribution of Firing Rates Across Electrode Array", 
             fontsize=12, fontweight="bold", pad=16, color="#1A202C")
ax.set_xlabel("Probe X Position (µm)", fontsize=11, labelpad=8, color="#2D3748")
ax.set_ylabel("Probe Y Position (µm)", fontsize=11, labelpad=8, color="#2D3748")

cbar = fig.colorbar(im, ax=ax, pad=0.04, fraction=0.046)
cbar.set_label("Total Firing Rate (Hz)", fontsize=10, labelpad=10, color="#2D3748")
cbar.ax.tick_params(labelsize=9, colors="#4A5568")
cbar.solids.set_edgecolor("face")

ax.tick_params(labelsize=10, colors="#4A5568")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#CBD5E0")
ax.spines["bottom"].set_color("#CBD5E0")
ax.set_facecolor("#F7FAFC")


plt.tight_layout()

output_path = os.path.join(folder_path, "firing_rate_spatial_topomap.png")
plt.savefig(output_path, dpi=300, facecolor='white')
print(f"Success! Spatial topomap saved to: {output_path}")
plt.show()