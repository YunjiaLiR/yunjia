import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt


folder_paths = [
    '20260303','20260302'
]

fs = 30000 
max_lag_ms = 40.0   # Look within a 40 ms window post-spike
bin_size_ms = 0.5   # 0.5 ms bins for clean temporal resolution
bins = np.arange(0, max_lag_ms + bin_size_ms, bin_size_ms)

# Master list to hold rise times from all neurons across all subjects
all_neuron_rise_times = []
total_good_clusters = 0

# ==========================================
# 2. PROCESS EACH FOLDER
# ==========================================
for folder_path in folder_paths:
    print(f"\n--- Processing folder: {folder_path} ---")
    
    # Check if directory exists before trying to load
    if not os.path.exists(folder_path):
        print(f"Warning: Folder '{folder_path}' not found. Skipping.")
        continue

    try:
        # Load metadata and spike files for this specific folder
        spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
        spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
        cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')
        metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)
    except Exception as e:
        print(f"Error loading files in {folder_path}: {e}. Skipping.")
        continue

    # Filter for good clusters
    keep_mask = (
        (metrics["isi_violations_ratio"] < 0.5) &
        (metrics["amplitude_cutoff"] < 0.1) &
        (metrics["presence_ratio"] > 0.8) & (metrics["amplitude_median"].abs() > 15)
    )

    good_clusters = metrics[keep_mask].index.values
    total_good_clusters += len(good_clusters)
    print(f"Identified {len(good_clusters)} 'good' single neurons in this folder.")

    # Calculate ACG rise times for the current folder
    folder_rise_times_count = 0
    for cluster_id in good_clusters:
        # Isolate spike times (in samples) for this cell and convert to milliseconds
        unit_spikes = spike_times[spike_clusters == cluster_id]
        
        if len(unit_spikes) < 50:
            continue  # Skip neurons with too few spikes
            
        unit_spikes_ms = (unit_spikes / fs) * 1000.0
        
        # Fast vectorized calculation of relative spike-to-spike time differences (lags)
        lags = []
        for shift in range(1, 100):  # Check up to 100 consecutive spikes forward
            diffs = unit_spikes_ms[shift:] - unit_spikes_ms[:-shift]
            valid_diffs = diffs[diffs <= max_lag_ms]
            if len(valid_diffs) == 0:
                break
            lags.extend(valid_diffs)
            
        if not lags:
            continue
            
        # Calculate the raw histogram counts for the auto-correlogram
        counts, _ = np.histogram(lags, bins=bins)
        
        # Apply a gentle median filter to smooth out random noise spikes
        smoothed_counts = medfilt(counts, kernel_size=3)
        
        # Find the time bin where the ACG reaches its first peak
        search_limit = int(25.0 / bin_size_ms)
        peak_bin_idx = np.argmax(smoothed_counts[:search_limit])
        
        # Calculate rise time as the center of that peak bin
        rise_time_ms = (peak_bin_idx * bin_size_ms) + (bin_size_ms / 2.0)
        
        # Append to the master list
        all_neuron_rise_times.append(rise_time_ms)
        folder_rise_times_count += 1
        
    print(f"Successfully computed ACG rise times for {folder_rise_times_count} neurons in {folder_path}.")

# ==========================================
# 3. PLOT COMBINED ACG RISE TIME DISTRIBUTION
# ==========================================
print("\n==========================================")
print(f"Total folders processed: {len(folder_paths)}")
print(f"Total 'good' neurons found: {total_good_clusters}")
print(f"Total neurons successfully computed for plot: {len(all_neuron_rise_times)}")
print("==========================================\n")

if len(all_neuron_rise_times) == 0:
    print("No valid rise times were calculated across any folders. Exiting without plotting.")
else:
    # Convert master list to array for calculations
    neuron_rise_times = np.array(all_neuron_rise_times)
    
    # Calculate accurate population metrics across all aggregated single neurons
    overall_median = np.median(neuron_rise_times)
    q25 = np.percentile(neuron_rise_times, 25)
    q75 = np.percentile(neuron_rise_times, 75)

    # Initialize the figure with spacious dimensions
    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # Subtle background grid lines for y-axis count readability
    ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CBD5E0', zorder=1)

    # Plot the distribution histogram (Count vs Rise Time) with spacing
    counts, bins_hist, patches = ax.hist(
        neuron_rise_times, 
        bins=15,             # Slightly increased bins since you have more pooled data
        color="#2B4C7E",     # Sleek Slate Blue
        edgecolor="none",    
        rwidth=0.85,         # Clean spacing between bars
        zorder=3
    )

    # Vertical dashed line at the median rise time (Crisp Crimson Red)
    ax.axvline(
        overall_median, 
        color="#E53E3E", 
        linestyle="--", 
        linewidth=2, 
        zorder=4,
        label=f"Median: {overall_median:.1f} ms"
    )

    # Soft matching blue background span denoting the Interquartile Range (IQR)
    ax.axvspan(
        q25, 
        q75, 
        color="#3182CE", 
        alpha=0.12,          # High transparency for background subtlety
        zorder=2,
        label=f"IQR ({q25:.1f} - {q75:.1f} ms)"
    )

    # Typography and Label Polish
    ax.set_title(f"Pooled Single Neuron ACG Rise Time Distribution\n(N = {len(neuron_rise_times)} Units across {len(folder_paths)} Subjects)", 
                 fontsize=12, fontweight="bold", pad=14, color="#1A202C")
    ax.set_xlabel("ACG Rise Time (ms)", fontsize=11, labelpad=8, color="#2D3748")
    ax.set_ylabel("Neuron Counts", fontsize=11, labelpad=8, color="#2D3748")

    # Dynamic axis limits with clean boundary padding
    x_padding = (max(neuron_rise_times) - min(neuron_rise_times)) * 0.15 if len(neuron_rise_times) > 1 else 2
    ax.set_xlim(max(0, min(neuron_rise_times) - x_padding), max(neuron_rise_times) + x_padding)
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

    # Save output (saving to the current working directory since it spans multiple folders)
    output_path = "pooled_neurons_acg_rise_time_distribution.png"
    plt.savefig(output_path, dpi=300, facecolor='white')
    print(f"Success! Pooled ACG rise time distribution plot saved to: {output_path}")
    plt.show()