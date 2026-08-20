import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


folder_paths = [
    '20260303','20260302'
    # Add more paths as needed
]

fs = 30000          
n_channels = 128    
dtype = 'int16'     

# Define the high-pass filter from your waveform script (only needs to be defined once)
# b, a = butter(3, 300 / (fs / 2), btype='high')
b, a = butter(3,[500 / (fs / 2), 3000 / (fs / 2)],btype='bandpass')
waveform_samples = int(0.002 * fs) # 2 ms window

# Master list to hold durations from all neurons across all subjects
all_neuron_durations = []
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

    print("Extracting raw waveforms to calculate physical durations...")
    folder_duration_count = 0

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

        best_trace = mean_wf[phy_channel, :]
        
        # Find the index of the depolarization trough (minimum voltage)
        trough_idx = np.argmin(best_trace)
        
        # Find the index of the subsequent repolarization peak (maximum voltage after trough)
        if trough_idx < len(best_trace) - 1:
            peak_idx_relative = np.argmax(best_trace[trough_idx:])
            peak_idx = trough_idx + peak_idx_relative
            
            # Calculate sample difference and convert to milliseconds (ms)
            duration_samples = peak_idx - trough_idx
            duration_ms = (duration_samples / fs) * 1000
            
            # Append to master list
            all_neuron_durations.append(duration_ms)
            folder_duration_count += 1

    print(f"Successfully computed durations for {folder_duration_count} neurons in {folder_path}.")
    
    # IMPORTANT: Free up memory before the next loop
    del memmap_data

# ==========================================
# 3. PLOT COMBINED TROUGH-TO-PEAK DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_durations)}")
print("==========================================\n")

if len(all_neuron_durations) == 0:
    print("No valid durations were calculated across any folders. Exiting without plotting.")
else:
    neuron_durations = np.array(all_neuron_durations)
    
    # Calculate accurate population metrics across pooled single neurons
    overall_median = np.median(neuron_durations)
    q25 = np.percentile(neuron_durations, 25)
    q75 = np.percentile(neuron_durations, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram (Count vs Duration) with spacing
    counts, bins, patches = ax.hist(
        neuron_durations, 
        bins=15,             # Increased bins slightly for pooled data
        color="#2B4C7E",     # Sleek Slate Blue
        edgecolor="none",    
        rwidth=0.85,         # Clean spacing between bars
        zorder=3
    )

    # Vertical dashed line at the median duration (Crisp Crimson Red)
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median: {overall_median:.2f} ms"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          # High transparency for background subtlety
        zorder=2,
        label=f"IQR ({q25:.2f} - {q75:.2f} ms)"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron Trough-to-Peak Distribution\n(N = {len(neuron_durations)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("Trough-to-Peak Duration (ms)", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic axis limits with clean boundary padding
    x_padding = (max(neuron_durations) - min(neuron_durations)) * 0.15 if len(neuron_durations) > 1 else 0.1
    ax.set_xlim(max(0, min(neuron_durations) - x_padding), max(neuron_durations) + x_padding)
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
    output_path = "pooled_neurons_trough_to_peak_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Pooled trough-to-peak distribution plot saved to: {output_path}")
    plt.show()