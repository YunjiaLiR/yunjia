import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ==========================================
# 1. CONFIGURATION & FILE PATHS
# ==========================================
# Provide a list of all your subject/session folders here
folder_paths = [
    '20260303','20260302'
]

fs = 30000          
n_channels = 128    
dtype = 'int16'     
bit_to_uv = 0.25   # Blackrock conversion factor (optional for SNR, but good to keep)

# Define the high-pass filter from your waveform script (only needs to be defined once)
b, a = butter(3, 300 / (fs / 2), btype='high')
waveform_samples = int(0.002 * fs) # 2 ms window

# Master list to hold SNR values from all neurons across all subjects
all_neuron_snrs = []
total_good_clusters = 0

for folder_path in folder_paths:
    print(f"\n--- Processing folder: {folder_path} ---")
    
    # Assume binary file is named matching the folder (e.g., '20260303_data.bin')
    binary_file = f"{folder_path}_data.bin"
    
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Skipping.")
        continue
    if not os.path.exists(binary_file):
        print(f"Warning: Binary file '{binary_file}' not found. Skipping.")
        continue

    try:
        # Load metadata and spike files
        spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
        spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
        metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)
        cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'),sep='\t')
    except Exception as e:
        print(f"Error loading metadata in {folder_path}: {e}. Skipping.")
        continue

    # Filter for good clusters
    keep_mask = (
        (metrics["isi_violations_ratio"] < 0.5) &
        (metrics["amplitude_cutoff"] < 0.1) &
        (metrics["presence_ratio"] > 0.8) & (metrics["amplitude_median"].abs() > 15)
    )

    good_clusters = metrics[keep_mask].index.values
    total_good_clusters += len(good_clusters)
    print(f"Identified {len(good_clusters)} 'good' single neurons from Phy labels.")

    # Initialize binary file reading via memmap
    file_bytes = os.path.getsize(binary_file)
    bytes_per_data_point = np.dtype(dtype).itemsize 
    total_samples = file_bytes // (n_channels * bytes_per_data_point)
    memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_samples, n_channels))

    print("Extracting waveforms and calculating channel noise floors...")
    folder_snr_count = 0

    for cluster_id in good_clusters:
        unit_spikes = spike_times[spike_clusters == cluster_id]
        
        # Subsample spikes to keep calculation fast while remaining accurate
        if len(unit_spikes) > 300:
            np.random.shuffle(unit_spikes)
            spikes_to_use = unit_spikes[:300]
        elif len(unit_spikes) < 20:
            continue  # Skip units with insufficient spikes
        else:
            spikes_to_use = unit_spikes

        snippets = []
        for spk in spikes_to_use:
            start_idx = spk - (waveform_samples // 2)
            end_idx = spk + (waveform_samples // 2)
            
            if start_idx > 0 and end_idx < memmap_data.shape[0]:
                snippets.append(memmap_data[start_idx:end_idx, :].T) 
                
        if not snippets:
            continue
            
        snippets = np.array(snippets)
        snippets_filtered = filtfilt(b, a, snippets, axis=-1)
        
        # Calculate the clean mean waveform across channels
        mean_wf = np.mean(snippets_filtered, axis=0)

        row = cluster_info.loc[cluster_info['cluster_id'] == cluster_id]

        if row.empty:
            print(f"Warning: Unit {cluster_id} not found in cluster_info.tsv")
            continue

        phy_channel = int(row.iloc[0]['ch'])

        # Signal on Phy channel
        max_ptp_raw = np.ptp(mean_wf[phy_channel, :])

        # Noise from the SAME Phy channel
        noise_samples = min(300000, total_samples)
        raw_noise_chunk = memmap_data[:noise_samples, phy_channel]
        filtered_noise_chunk = filtfilt(b, a, raw_noise_chunk)
        
        # Standard deviation of the filtered background trace represents the noise floor
        noise_std_raw = np.std(filtered_noise_chunk)
        
        # 3. Calculate SNR ratio (bit_to_uv cancels out naturally)
        if noise_std_raw > 0:
            snr = max_ptp_raw / noise_std_raw
            all_neuron_snrs.append(snr)
            folder_snr_count += 1

    print(f"Successfully computed SNR for {folder_snr_count} neurons in {folder_path}.")
    
    # IMPORTANT: Free up memory before the next loop
    del memmap_data

# ==========================================
# 3. PLOT COMBINED SNR DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_snrs)}")
print("==========================================\n")

if len(all_neuron_snrs) == 0:
    print("No valid SNRs were calculated across any folders. Exiting without plotting.")
else:
    neuron_snrs = np.array(all_neuron_snrs)
    
    # Calculate accurate population metrics across pooled single neurons
    overall_median = np.median(neuron_snrs)
    q25 = np.percentile(neuron_snrs, 25)
    q75 = np.percentile(neuron_snrs, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram (Count vs SNR) with spacing
    counts, bins, patches = ax.hist(
        neuron_snrs, 
        bins=15,             # Increased bins slightly for pooled data
        color="#2B4C7E",     # Sleek Slate Blue
        edgecolor="none",    
        rwidth=0.85,         # Clean spacing between bars
        zorder=3
    )

    # Vertical dashed line at the median SNR (Crisp Crimson Red)
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median SNR: {overall_median:.1f}"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          # High transparency for background subtlety
        zorder=2,
        label=f"IQR ({q25:.1f} - {q75:.1f})"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron SNR Distribution\n(N = {len(neuron_snrs)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("Signal-to-Noise Ratio (SNR)", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic axis limits with clean boundary padding
    x_padding = (max(neuron_snrs) - min(neuron_snrs)) * 0.15 if len(neuron_snrs) > 1 else 2
    ax.set_xlim(max(0, min(neuron_snrs) - x_padding), max(neuron_snrs) + x_padding)
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
    output_path = "pooled_neurons_snr_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Pooled SNR distribution plot saved to: {output_path}")
    plt.show()