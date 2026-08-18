import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

# Define the high-pass filter (only needs to be defined once)
b, a = butter(3, 300 / (fs / 2), btype='high')
waveform_samples = int(0.002 * fs) # 2 ms window

# Master list to hold ratios from all neurons across all subjects
all_neuron_ratios = []
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
        # Load metadata and spike files
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

    # Initialize binary file reading via memmap
    file_bytes = os.path.getsize(binary_file)
    bytes_per_data_point = np.dtype(dtype).itemsize 
    total_samples = file_bytes // (n_channels * bytes_per_data_point)
    memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_samples, n_channels))

    print("Extracting raw waveforms to calculate peak-trough ratios...")
    folder_ratio_count = 0

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
        
        # Identify the dominant peak channel using your script's logic
        peak_channel = np.argmax(np.max(np.abs(mean_wf), axis=1))
        
        # Isolate the 1D mean trace on that dominant channel
        best_trace = mean_wf[peak_channel, :]
        
        # Find the index and value of the negative depolarization trough
        trough_idx = np.argmin(best_trace)
        trough_val = best_trace[trough_idx]
        
        # Find the maximum positive peak value strictly occurring after the trough
        if trough_idx < len(best_trace) - 1:
            peak_val = np.max(best_trace[trough_idx:])
            
            # Calculate the absolute ratio (Peak Amplitude / Trough Amplitude)
            if abs(trough_val) > 0:
                pt_ratio = abs(peak_val) / abs(trough_val)
                all_neuron_ratios.append(pt_ratio)
                folder_ratio_count += 1

    print(f"Successfully computed peak-trough ratios for {folder_ratio_count} neurons in {folder_path}.")
    
    # IMPORTANT: Free up memory before the next loop
    del memmap_data

# ==========================================
# 3. PLOT COMBINED PEAK-TROUGH RATIO DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_ratios)}")
print("==========================================\n")

if len(all_neuron_ratios) == 0:
    print("No valid peak-trough ratios were calculated across any folders. Exiting without plotting.")
else:
    neuron_ratios = np.array(all_neuron_ratios)
    
    # Calculate accurate population metrics across pooled single neurons
    overall_median = np.median(neuron_ratios)
    q25 = np.percentile(neuron_ratios, 25)
    q75 = np.percentile(neuron_ratios, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram with modern coloring and gaps
    counts, bins, patches = ax.hist(
        neuron_ratios, 
        bins=15,             # Increased bins slightly for pooled data
        color="#2B4C7E",     # Sleek Slate Blue
        edgecolor="none",    
        rwidth=0.85,         # Clean spacing between bars
        zorder=3
    )

    # Vertical dashed line at the median ratio (Crisp Crimson Red)
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median: {overall_median:.2f}"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          # High transparency for background subtlety
        zorder=2,
        label=f"IQR ({q25:.2f} - {q75:.2f})"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron Peak-Trough Ratio\n(N = {len(neuron_ratios)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("Peak-Trough Ratio", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic axis limits with clean boundary padding
    x_padding = (max(neuron_ratios) - min(neuron_ratios)) * 0.15 if len(neuron_ratios) > 1 else 0.2
    ax.set_xlim(max(0, min(neuron_ratios) - x_padding), max(neuron_ratios) + x_padding)
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
    output_path = "pooled_neurons_peak_trough_ratio_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Peak-trough ratio distribution plot saved to: {output_path}")
    plt.show()