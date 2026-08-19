import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import butter, filtfilt

binary_file = '20260804_data.bin'
fs = 30000          
n_channels = 128    
dtype = 'int16'     
bit_to_uv = 0.25   # blackrock

# DATA LOADING
print("Loading data...")
folder_path = '20260804'

spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')
metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)
channel_map = np.load(os.path.join(folder_path, 'channel_map.npy')).flatten()
channel_positions = np.load(os.path.join(folder_path, 'channel_positions.npy'))

print(f"Raw binary channels configured: {n_channels}")
print(f"Kilosort channel_map length: {len(channel_map)}")
print(f"Kilosort channel_positions length: {len(channel_positions)}")
print("Identity channel mapping:",
      np.array_equal(channel_map, np.arange(len(channel_map))))

keep_mask = ((metrics["isi_violations_ratio"] < 0.5) &(metrics["amplitude_cutoff"] < 0.1) &(metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 15) )#(metrics["amplitude_median"] > 10) 

good_clusters = metrics[keep_mask].index.values

file_bytes = os.path.getsize(binary_file)
bytes_per_data_point = np.dtype(dtype).itemsize 
total_samples = file_bytes // (n_channels * bytes_per_data_point)

memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_samples, n_channels))

# High-pass filter 
b, a = butter(3, 300 / (fs / 2), btype='high')

# EXTRACT WAVEFORMS & CALCULATE SPREAD
print(f"Extracting snippets for {len(good_clusters)} good units...")

extracted_units = []
waveform_samples = int(0.002 * fs) # 2 ms window
time_ms = np.linspace(0, 2.0, waveform_samples)

extracted_units = []

for cluster_id in good_clusters:
    unit_spikes = spike_times[spike_clusters == cluster_id]
    
    if len(unit_spikes) > 500:
        np.random.shuffle(unit_spikes)
        spikes_to_use = unit_spikes[:500]
    elif len(unit_spikes) < 50:
        continue
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

    # IMPORTANT:
    # Do not high-pass filter each 2-ms snippet independently.
    # Filtering such a short segment can create strong edge artifacts.
    # For debugging / waveform display, average the extracted raw snippets directly.
    mean_wf = np.mean(snippets, axis=0)
    std_wf = np.std(snippets, axis=0)
  
    # Get canonical best channel from Phy / Kilosort
    row = cluster_info.loc[cluster_info['cluster_id'] == cluster_id]

    if row.empty:
        print(f"Warning: Unit {cluster_id} not found in cluster_info.tsv")
        continue

    phy_channel = int(row.iloc[0]['ch'])

    # Use Phy/Kilosort as the canonical channel assignment.
    #
    # IMPORTANT:
    # cluster_info.tsv 'ch' may represent either the mapped raw channel ID
    # or the Kilosort channel index, depending on how the output was generated.
    # We therefore try to use it directly if it is a valid raw-binary column.
    # If not, map through channel_map.npy.
    if 0 <= phy_channel < n_channels:
        raw_channel = phy_channel
    elif 0 <= phy_channel < len(channel_map):
        raw_channel = int(channel_map[phy_channel])
    else:
        print(
            f"Warning: Unit {cluster_id}: Phy channel {phy_channel} is invalid "
            f"for both raw binary ({n_channels} ch) and channel_map ({len(channel_map)} ch)."
        )
        continue

    # QC: if phy_channel can index channel_map, show what that mapping would be.
    mapped_candidate = (
        int(channel_map[phy_channel])
        if 0 <= phy_channel < len(channel_map)
        else None
    )

    # Calculate channel spread relative to the canonical raw channel
    ptp_all = np.ptp(mean_wf, axis=1)
    max_ptp = ptp_all[raw_channel]
    spread = np.sum(ptp_all > 0.2 * max_ptp)

    # Extract waveform from canonical Phy/Kilosort channel
    mean_trace = mean_wf[raw_channel, :] * bit_to_uv
    std_trace = std_wf[raw_channel, :] * bit_to_uv

    # Compare waveform peak with SpikeInterface amplitude_median
    plot_peak = float(np.max(np.abs(mean_trace)))
    si_amp = float(abs(metrics.loc[cluster_id, "amplitude_median"]))

    print(
        f"Unit {cluster_id:4d} | "
        f"Phy ch {phy_channel:3d} | "
        f"mapped candidate {str(mapped_candidate):>4} | "
        f"raw ch used {raw_channel:3d} | "
        f"SI median {si_amp:7.2f} uV | "
        f"mean waveform peak {plot_peak:7.2f} uV"
    )
            
    extracted_units.append({
        'unit_id': cluster_id,
        'channel': phy_channel,      # canonical Phy/Kilosort label shown on figure
        'raw_channel': raw_channel,  # actual binary column used
        'mean': mean_trace,
        'std': std_trace,
        'spread': spread,
        'plot_peak': plot_peak,
        'si_amp': si_amp
    })

# ==========================================
# QC SUMMARY
# ==========================================
if extracted_units:
    largest = max(extracted_units, key=lambda x: x["plot_peak"])
    print("\n=== LARGEST PLOTTED WAVEFORM ===")
    print(f"Unit: {largest['unit_id']}")
    print(f"Phy/Kilosort channel label: {largest['channel']}")
    print(f"Raw binary channel used: {largest['raw_channel']}")
    print(f"Plotted mean-waveform peak: {largest['plot_peak']:.2f} uV")
    print(f"SpikeInterface amplitude_median: {largest['si_amp']:.2f} uV")
else:
    print("No units were extracted.")

# ==========================================
# 2. PLOTTING
# ==========================================
print("Plotting waveforms...")

n_units = len(extracted_units)
cols = 6
rows = max(2, math.ceil((n_units + 4) / cols))

if not extracted_units:
    raise RuntimeError("No units available for plotting after filtering/extraction.")

all_uv = np.concatenate([u['mean'] for u in extracted_units])
global_max = np.max(np.abs(all_uv)) * 1.1

print(f"Global plotting limit: +/-{global_max:.2f} uV")

fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))
gs = gridspec.GridSpec(rows, cols, wspace=0.1, hspace=0.6)

colors = plt.cm.tab20.colors
unit_idx = 0

for r in range(rows):
    for c in range(cols):
        # Skip bottom-right 2x2 area for the bar chart
        if r >= rows - 2 and c >= cols - 2:
            continue
            
        if unit_idx < n_units:
            ax = fig.add_subplot(gs[r, c])
            unit = extracted_units[unit_idx]
            color = colors[unit_idx % len(colors)]
            
            mean_line = unit['mean']
            std_line = unit['std']
            
            # 1. Shaded STD band (Shadow)
            ax.fill_between(
                time_ms,
                mean_line - std_line,
                mean_line + std_line,
                color=color,
                alpha=0.3,
                linewidth=0
            )
            
            # 2. Solid Mean Trace
            ax.plot(time_ms, mean_line, color=color, alpha=1.0, linewidth=2)
            
            # Subplot formatting
            ax.axis('off')
            ax.set_ylim(-global_max, global_max)
            ax.set_xlim(0, max(time_ms))
            
            channel_text = f"Chn #{unit['channel']}"
            if unit['raw_channel'] != unit['channel']:
                channel_text += f" (raw {unit['raw_channel']})"

            ax.text(
                0.0, 1.02,
                f"{channel_text} | Unit #{unit['unit_id']}",
                transform=ax.transAxes,
                fontsize=10,
                va='bottom',
                ha='left',
                color='black'
            )
            
            unit_idx += 1

# Draw Custom L-Shaped Scale Bar
ax_scale = fig.add_subplot(gs[rows-1, 0])
ax_scale.axis('off')
ax_scale.set_ylim(-global_max, global_max)
ax_scale.set_xlim(0, max(time_ms))

x_start = 0.2
y_start = -global_max * 1.5

ax_scale.plot([x_start, x_start], [y_start, y_start + 20], color='black', lw=2, clip_on=False)
ax_scale.plot([x_start, x_start + 1.0], [y_start, y_start], color='black', lw=2, clip_on=False)

ax_scale.text(x_start - 0.2, y_start + 10, '20 µV', rotation=90, va='center', ha='center', fontsize=14)
ax_scale.text(x_start + 0.5, y_start - (global_max * 0.15), '1.0 ms', va='top', ha='center', fontsize=14)

# Draw Spread Histogram
ax_bar = fig.add_subplot(gs[rows-2:rows, cols-2:cols])

spreads = [u['spread'] for u in extracted_units]
if spreads:
    max_spread = max(spreads)
    bins = np.arange(1, max(8, max_spread + 2)) - 0.5
    
    ax_bar.hist(spreads, bins=bins, color='silver', edgecolor='black', rwidth=0.5)
    ax_bar.set_xlim(0.5, 6.5)
    ax_bar.set_xticks(range(1, 7))
    
ax_bar.set_title("Unit counts", fontsize=14)
ax_bar.set_xlabel("# of spreaded chns", fontsize=12)
ax_bar.tick_params(axis='both', labelsize=10)

ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.08)
output_filename = 'kilosort_waveforms_publication_layout.png'
plt.savefig(output_filename, dpi=300, facecolor='white')
print(f"Success! Saved to {output_filename}")

plt.show()