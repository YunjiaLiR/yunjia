import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

binary_file = '20260302_data.bin'
fs = 30000 
n_channels = 128 
dtype = 'int16'  
bit_to_uv = 0.25  

exclude_channels = [31, 32, 64, 65, 95, 96, 127, 128]


folder_path = '20260302'
spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')
metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)

keep_mask = ((metrics["isi_violations_ratio"] < 0.5) &(metrics["amplitude_cutoff"] < 0.1) &(metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 15))

good_clusters = metrics[keep_mask].index.values

# Load 0.5-second snippet
start_sec = 0       
duration_sec = 0.5    
start_sample = int(start_sec * fs)
n_samples = int(duration_sec * fs)

file_bytes = os.path.getsize(binary_file)
bytes_per_data_point = np.dtype(dtype).itemsize 
total_records = file_bytes // (n_channels * bytes_per_data_point)

memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_records, n_channels))
raw_segment = memmap_data[start_sample:start_sample + n_samples, :].T 

# b, a = butter(3, 300 / (fs / 2), btype='high')
b, a = butter(3,[500 / (fs / 2), 3000 / (fs / 2)],btype='bandpass')
highpass_segment = filtfilt(b, a, raw_segment, axis=1)

waveforms_dict = {}

waveform_samples = int(0.002 * fs) # 2 ms window

print('Calculating average waveforms for good neurons...')
for cluster in good_clusters:
    unit_spikes = spike_times[spike_clusters == cluster]
    if len(unit_spikes) < 50: continue
    
    if len(unit_spikes) > 500:
        np.random.shuffle(unit_spikes)
        spikes_to_use = unit_spikes[:500]
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
    mean_wf = np.mean(snippets_filtered, axis=0)
  
    row = cluster_info.loc[cluster_info['cluster_id'] == cluster]

    if row.empty:
        print(f"Warning: Unit {cluster} not found in cluster_info.tsv")
        continue

    phy_channel = int(row.iloc[0]['ch'])

    if phy_channel in exclude_channels:
        continue

    waveforms_dict[cluster] = {
        'channel': phy_channel,
        'mean': mean_wf[phy_channel, :],
    }

# PLOTTING
fig = plt.figure(figsize=(16, 20))

gs = gridspec.GridSpec(1, 7, width_ratios=[0.2, 1, 1, 0.3, 1, 1, 0.3], wspace=0.1)

axes = {
    'probe': fig.add_subplot(gs[0]),
    'raw_L': fig.add_subplot(gs[1]),
    'hp_L':  fig.add_subplot(gs[2]),
    'wf_L':  fig.add_subplot(gs[3]),
    'raw_R': fig.add_subplot(gs[4]),
    'hp_R':  fig.add_subplot(gs[5]),
    'wf_R':  fig.add_subplot(gs[6])
}

time_axis = np.linspace(0, duration_sec, n_samples)
wf_time_axis = np.linspace(0, 2.0, waveform_samples)

raw_scale = 1.0 / (np.max(np.abs(raw_segment)) * 0.15)       
highpass_scale = 1.0 / (np.max(np.abs(highpass_segment)) * 0.8) 

uv_per_row = 70.0 
wf_scale = 1.0 / uv_per_row

n_half = n_channels // 2  

for ch in range(n_channels):
    if ch in exclude_channels:
        continue  
        
    if ch < n_half:
        ax_raw, ax_hp = axes['raw_L'], axes['hp_L']
        offset = ch
    else:
        ax_raw, ax_hp = axes['raw_R'], axes['hp_R']
        offset = ch - n_half  
        
    ax_raw.plot(time_axis, (raw_segment[ch, :] * raw_scale) + offset, color='black', linewidth=0.5)
    ax_hp.plot(time_axis, (highpass_segment[ch, :] * highpass_scale) + offset, color='black', linewidth=0.5)

    ax_raw.text(-0.02, offset, str(ch), transform=ax_raw.get_yaxis_transform(), 
                ha='right', va='center', fontsize=8, color='black')

# PLOT WAVEFORMS 
colors = plt.cm.tab20.colors 
channel_counts = {} 

for i, (cluster_id, data) in enumerate(waveforms_dict.items()):
    ch = data['channel']
    
    if ch in exclude_channels: 
        continue
        
    mean_uv = data['mean'] * bit_to_uv
    unit_color = colors[i % len(colors)]
    
    if ch < n_half:
        ax_wf = axes['wf_L']
        base_offset = ch
    else:
        ax_wf = axes['wf_R']
        base_offset = ch - n_half
        
    if ch not in channel_counts:
        channel_counts[ch] = 0
    
    count = channel_counts[ch]
    if count == 0: shift = 0.0      
    elif count == 1: shift = 0.75     # Increased from 0.35
    elif count == 2: shift = -0.75    # Increased from -0.35
    elif count == 3: shift = 1.90     # Increased from 0.70
    else: shift = -1.70               # Increased from -0.70  
        
    channel_counts[ch] += 1
    final_offset = base_offset + shift

    ax_wf.plot(wf_time_axis, (mean_uv * wf_scale) + final_offset, color=unit_color, linewidth=1.5)


#DRAW PROBE SCHEMATIC 
axes['probe'].set_xlim(0, 1)
axes['probe'].add_patch(patches.Rectangle((0.2, -1), 0.6, n_half + 2, color='#fdeebf', zorder=0))

for ch in range(n_half):
    if ch in exclude_channels:
        continue 
        
    circle = patches.Circle((0.8, ch), radius=0.3, color='#e99b3b', zorder=1)
    axes['probe'].add_patch(circle)
    axes['probe'].text(0.1, ch, str(ch), va='center', ha='right', fontsize=7)


# DRAW SCALE BARS (UNIT LENGTHS)
scale_y = n_half + 2.0 

# 1. Raw Data Scale Bar (50 ms) - X axis goes from 0 to 0.5 seconds
axes['raw_L'].plot([0.1, 0.15], [scale_y, scale_y], color='black', lw=3, clip_on=False)
axes['raw_L'].text(0.125, scale_y + 0.6, '50 ms', ha='center', va='top', fontsize=10)

# 2. Highpass Data Scale Bar (50 ms)
axes['hp_L'].plot([0.1, 0.15], [scale_y, scale_y], color='black', lw=3, clip_on=False)
axes['hp_L'].text(0.125, scale_y + 0.6, '50 ms', ha='center', va='top', fontsize=10)

# 3. Waveform Scale Bar (1 ms horizontal, 50 uV vertical)
wf_bar_x = 0.0 # start at 0 ms
axes['wf_L'].plot([wf_bar_x, wf_bar_x + 1.0], [scale_y, scale_y], color='black', lw=3, clip_on=False)
axes['wf_L'].text(wf_bar_x + 0.5, scale_y + 0.6, '1 ms', ha='center', va='top', fontsize=10)

# Calculate height for 50 uV based on our scale
uV_height = 50.0 / uv_per_row
axes['wf_L'].plot([wf_bar_x, wf_bar_x], [scale_y, scale_y - uV_height], color='black', lw=3, clip_on=False)
axes['wf_L'].text(wf_bar_x - 0.2, scale_y - (uV_height/2), r'50 $\mu$V', ha='right', va='center', fontsize=10)


# Add Column Titles
axes['raw_L'].set_title('Raw', fontsize=14, y=-0.02)
axes['hp_L'].set_title('Highpass', fontsize=14, y=-0.02)
axes['raw_R'].set_title('Raw', fontsize=14, y=-0.02)
axes['hp_R'].set_title('Highpass', fontsize=14, y=-0.02)

# Apply unified styling to all subplots
for name, ax in axes.items():
    ax.axis('off')  
    # --- NEW: Extended the bottom ylim to make room for scale bars ---
    ax.set_ylim(n_half + 4, -2) 

plt.tight_layout()

# Save and run
output_filename = 'kilosort_split_layout.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Success! Publication layout saved to {output_filename}")

plt.show()
plt.close()