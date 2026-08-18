import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Provide a list of all your subject/session folders here
folder_paths = [
    '20260303','20260302'
]

fs = 30000          
n_channels = 128    
dtype = 'int16'     

# Master list to hold firing rates from all neurons across all subjects
all_neuron_firing_rates = []
total_good_clusters = 0

# ==========================================
# 2. PROCESS EACH FOLDER
# ==========================================
for folder_path in folder_paths:
    print(f"\n--- Processing folder: {folder_path} ---")
    
    # Assume binary file is named matching the folder (e.g., '20260303_data.bin')
    # If your binary files are inside the folders, change this to: os.path.join(folder_path, 'data.bin')
    binary_file = f"{folder_path}_data.bin"
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Skipping.")
        continue
    if not os.path.exists(binary_file):
        print(f"Warning: Binary file '{binary_file}' not found. Skipping.")
        continue

    try:
        # Load metadata and spike files for this specific folder
        spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
        spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
        metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)
    except Exception as e:
        print(f"Error loading metadata in {folder_path}: {e}. Skipping.")
        continue

    # Filter for good clusters
    keep_mask = (
        (metrics["isi_violations_ratio"] < 0.5) &
        (metrics["amplitude_cutoff"] < 0.1) &
        (metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 15)
    )

    good_clusters = metrics[keep_mask].index.values
    total_good_clusters += len(good_clusters)
    print(f"Identified {len(good_clusters)} 'good' single neurons from Phy labels.")

    # Determine the exact recording duration from this specific binary file
    file_bytes = os.path.getsize(binary_file)
    bytes_per_data_point = np.dtype(dtype).itemsize 
    total_samples = file_bytes // (n_channels * bytes_per_data_point)
    duration_seconds = total_samples / fs
    print(f"Recording duration for {folder_path}: {duration_seconds:.2f} seconds ({duration_seconds/60:.2f} minutes)")

    # Calculate firing rates for the current folder
    folder_firing_rate_count = 0
    for cluster_id in good_clusters:
        # Count the total number of spikes assigned to this cluster
        n_spikes = np.sum(spike_clusters == cluster_id)
        
        # Firing rate = Total Spikes / Total Duration (Hz)
        firing_rate_hz = n_spikes / duration_seconds
        
        # Append to master list
        all_neuron_firing_rates.append(firing_rate_hz)
        folder_firing_rate_count += 1

    print(f"Successfully computed firing rates for {folder_firing_rate_count} neurons in {folder_path}.")

# ==========================================
# 3. PLOT COMBINED FIRING RATE DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_firing_rates)}")
print("==========================================\n")

if len(all_neuron_firing_rates) == 0:
    print("No valid firing rates were calculated across any folders. Exiting without plotting.")
else:
    # Convert master list to array for calculations
    neuron_firing_rates = np.array(all_neuron_firing_rates)
    
    # Calculate population metrics across pooled single neurons
    overall_median = np.median(neuron_firing_rates)
    q25 = np.percentile(neuron_firing_rates, 25)
    q75 = np.percentile(neuron_firing_rates, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram (Count vs Firing Rate) with spacing
    counts, bins, patches = ax.hist(
        neuron_firing_rates, 
        bins=15,             # Increased slightly for pooled data
        color="#2B4C7E",     # Sleek Slate Blue
        edgecolor="none",    
        rwidth=0.85,         # Clean spacing between bars
        zorder=3
    )

    # Vertical dashed line at the median firing rate (Crisp Crimson Red)
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median: {overall_median:.2f} Hz"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          # High transparency for background subtlety
        zorder=2,
        label=f"IQR ({q25:.2f} - {q75:.2f} Hz)"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron Firing Rate Distribution\n(N = {len(neuron_firing_rates)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("Firing Rate (Hz)", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic x-axis and y-axis limits with clean boundary padding
    x_padding = (max(neuron_firing_rates) - min(neuron_firing_rates)) * 0.15 if len(neuron_firing_rates) > 1 else 2
    ax.set_xlim(max(0, min(neuron_firing_rates) - x_padding), max(neuron_firing_rates) + x_padding)
    ax.set_ylim(0, max(counts) * 1.15)

    # Classic inward-pointing ticks style
    ax.tick_params(direction="in", top=False, right=False, labelsize=10, colors="#4A5568")

    # Remove top and right boundary spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E0")
    ax.spines["bottom"].set_color("#CBD5E0")

    # Render a borderless, clean legend layout
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    plt.tight_layout()

    # Save output to root directory
    output_path = "pooled_neurons_firing_rate_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Firing rate distribution plot saved to: {output_path}")
    plt.show()