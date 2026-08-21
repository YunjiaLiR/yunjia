import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

folder_paths = [
    '20260303','20260302'
]

fs = 30000          
n_channels = 128    
dtype = 'int16'     
bit_to_uv = 0.25   # Blackrock conversion factor

#b, a = butter(3, 300 / (fs / 2), btype='high')
b, a = butter(3,[500 / (fs / 2), 3000 / (fs / 2)],btype='bandpass')
waveform_samples = int(0.002 * fs) # 2 ms window

# Master list to hold amplitudes from all neurons across all subjects
all_neuron_amplitudes = []
total_good_clusters = 0

for folder_path in folder_paths:
    print(f"\n--- Processing folder: {folder_path} ---")
    
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
        (metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 15)
    )

    good_clusters = metrics[keep_mask].index.values
    total_good_clusters += len(good_clusters)
    print(f"Identified {len(good_clusters)} 'good' single neurons from Phy labels.")

    # Initialize binary file reading via memmap
    file_bytes = os.path.getsize(binary_file)
    bytes_per_data_point = np.dtype(dtype).itemsize 
    total_samples = file_bytes // (n_channels * bytes_per_data_point)
    memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_samples, n_channels))

    print("Extracting raw snippets to calculate physical amplitudes...")
    folder_amplitudes_count = 0

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
        
        # Get Peak-to-Peak (PTP) amplitude for all channels
        ptp_all_channels = np.ptp(mean_wf, axis=1)
        
        row = cluster_info.loc[cluster_info['cluster_id'] == cluster_id]

        if row.empty:
            print(f"Warning: Unit {cluster_id} not found in cluster_info.tsv")
            continue

        phy_channel = int(row.iloc[0]['ch'])

        max_ptp_raw = np.ptp(mean_wf[phy_channel, :])
        max_ptp_uv = max_ptp_raw * bit_to_uv
        # Append to master list
        all_neuron_amplitudes.append(max_ptp_uv)
        folder_amplitudes_count += 1

    print(f"Successfully computed amplitudes for {folder_amplitudes_count} neurons in {folder_path}.")
    
    # Free up memory before the next loop
    del memmap_data

# ==========================================
# 3. PLOT COMBINED AMPLITUDE DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_amplitudes)}")
print("==========================================\n")

if len(all_neuron_amplitudes) == 0:
    print("No valid amplitudes were calculated across any folders. Exiting without plotting.")
else:
    neuron_amplitudes = np.array(all_neuron_amplitudes)
    
    # Calculate accurate population metrics across pooled single neurons
    overall_median = np.median(neuron_amplitudes)
    q25 = np.percentile(neuron_amplitudes, 25)
    q75 = np.percentile(neuron_amplitudes, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram with modern coloring and gaps
    counts, bins, patches = ax.hist(
        neuron_amplitudes, 
        bins=15,             # Increased bins for larger pooled data
        color="#2B4C7E",     
        edgecolor="none",    
        rwidth=0.85,         
        zorder=3
    )

    # Vertical dashed line at the true physical median
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median: {overall_median:.1f} µV"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          
        zorder=2,
        label=f"IQR ({q25:.1f} - {q75:.1f} µV)"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron Amplitude Distribution\n(N = {len(neuron_amplitudes)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("Neuron Amplitude (µV)", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic axis limits with clean boundaries
    x_padding = (max(neuron_amplitudes) - min(neuron_amplitudes)) * 0.15 if len(neuron_amplitudes) > 1 else 10
    ax.set_xlim(max(0, min(neuron_amplitudes) - x_padding), max(neuron_amplitudes) + x_padding)
    ax.set_ylim(0, max(counts) * 1.15)

    # Replicate classic inward-pointing ticks style
    ax.tick_params(direction="in", top=False, right=False, labelsize=10, colors="#4A5568")

    # Clean frame adjustments
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CBD5E0")
    ax.spines["bottom"].set_color("#CBD5E0")

    # Render a borderless, clean legend layout
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    plt.tight_layout()

    # Save output to the root directory
    output_path = "pooled_neurons_amplitude_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Improved pooled distribution plot saved to: {output_path}")
    plt.show()