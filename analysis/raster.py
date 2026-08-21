import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.patches as mpatches

# ==========================================
# 1. PATHS AND PARAMETERS 
# ==========================================
binary_file = '20260302_data.bin'
folder_path = '20260302'
fs = 30000          
n_channels = 128    

# Define the time window you want to visualize
start_time_sec = 0     
end_time_sec = 30      


# Define colors for your 6 new grouped shanks
shank_colors = {
    1: 'blue',   
    2: 'red',    
    3: 'green',  
    4: 'orange',
    5: 'purple',
    6: 'cyan'
}

# ==========================================
# 2. LOAD METADATA AND SPIKE FILES
# ==========================================
print("Loading metadata and spike files...")
spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')
metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)

spike_times_sec = spike_times / fs

# ==========================================
# 3. FILTER FOR "GOOD" SINGLE NEURONS
# ==========================================
keep_mask = (
    (metrics["isi_violations_ratio"] < 0.5) & 
    (metrics["amplitude_cutoff"] < 0.1) & 
    (metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 20)
)

original_good_clusters = metrics[keep_mask].index.values
num_good_neurons = len(original_good_clusters)
print(f"Identified {num_good_neurons} 'good' single neurons.")

# ==========================================
# 4. EXTRACT CHANNEL AND SHANK INFO
# ==========================================
id_col = 'cluster_id' if 'cluster_id' in cluster_info.columns else 'id'
ch_col = 'ch' if 'ch' in cluster_info.columns else 'channel'
sh_col = 'sh' if 'sh' in cluster_info.columns else 'shank'

if sh_col not in cluster_info.columns:
    print("Warning: No shank column found! Defaulting everything to Shank 1.")
    cluster_info['sh'] = 1 
    sh_col = 'sh'

# ==========================================
# 5. CALCULATE Y-OFFSETS & APPLY MAPPING
# ==========================================
channels_to_units = defaultdict(list)
for cid in original_good_clusters:
    row = cluster_info[cluster_info[id_col] == cid].iloc[0]
    ch = row[ch_col]
    channels_to_units[ch].append(cid)

unit_y_positions = {}
unit_colors = {}
unique_mapped_shanks_present = set() # To keep track of which shanks to put in the legend

for ch, units in channels_to_units.items():
    n_units = len(units)
    
    if n_units == 1:
        offsets = [0]
    else:
        offsets = np.linspace(-0.3, 0.3, n_units)
        
    for i, cid in enumerate(units):
        unit_y_positions[cid] = ch + offsets[i]
        
        # Get raw shank, apply mapping, and assign color
        row = cluster_info[cluster_info[id_col] == cid].iloc[0]
        raw_shank = int(row[sh_col])
        
        # Map it using the dictionary above (defaults to raw_shank if not found)
        # Ensure mapped_shank is an integer
        mapped_shank = raw_shank
        unique_mapped_shanks_present.add(mapped_shank)
        
        # Assign color based on the integer mapped shank
        unit_colors[cid] = shank_colors.get(mapped_shank, 'black')

# ==========================================
# 6. FILTER SPIKES AND MAP TO NEW Y-POSITIONS
# ==========================================
time_mask = (spike_times_sec >= start_time_sec) & (spike_times_sec <= end_time_sec)
cluster_mask = np.isin(spike_clusters, original_good_clusters)
final_mask = time_mask & cluster_mask

filtered_times = spike_times_sec[final_mask]
original_filtered_clusters = spike_clusters[final_mask]

mapped_y_positions = np.array([unit_y_positions[cid] for cid in original_filtered_clusters])
mapped_colors = np.array([unit_colors[cid] for cid in original_filtered_clusters])

# ==========================================
# 7. PLOT THE RASTER
# ==========================================
print(f"Plotting {len(filtered_times)} spikes...")
plt.figure(figsize=(12, 8))

plt.scatter(filtered_times, mapped_y_positions, c=mapped_colors, marker='|', s=20, linewidths=0.8)

plt.title(f"Spatial Raster Plot (Grouped by Shank) - {start_time_sec} to {end_time_sec} s")
plt.xlabel("Time (seconds)")
plt.ylabel("Electrode Channel")

plt.xlim(start_time_sec, end_time_sec)
plt.ylim(-1, n_channels) 

plt.grid(True, axis='y', linestyle='-', alpha=0.2) 
plt.grid(True, axis='x', linestyle='--', alpha=0.5)

# Create a custom legend using the MAPPED shanks
legend_handles = [
    mpatches.Patch(color=shank_colors[sh], label=f'Shank Group {sh}') 
    for sh in sorted(unique_mapped_shanks_present) if sh in shank_colors
]
if legend_handles:
    plt.legend(handles=legend_handles, loc='upper right', title="Probe Shanks")

plt.tight_layout()

# Save image
output_image_path = os.path.join(folder_path, "grouped_shank_raster.png")
plt.savefig(output_image_path, dpi=300)
plt.show()

print(f"Saved raster plot to: {output_image_path}")