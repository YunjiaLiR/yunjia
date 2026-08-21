import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt
import matplotlib.gridspec as gridspec

# --- CONFIGURATION ---
binary_file = '20260302_data.bin'
fs = 30000 
total_channels = 128  
dtype = 'int16'  
bit_to_uv = 0.25  

# Target channel range
selected_channels = list(range(0, 19))
n_selected = len(selected_channels)

exclude_channels = [] 

folder_path = '20260302'
spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()
cluster_info = pd.read_csv(os.path.join(folder_path, 'cluster_info.tsv'), sep='\t')
metrics = pd.read_csv(os.path.join(folder_path, 'all_quality_metrics.csv'), index_col=0)

keep_mask = ((metrics["isi_violations_ratio"] < 0.5) & 
             (metrics["amplitude_cutoff"] < 0.1) & 
             (metrics["presence_ratio"] > 0.8)& (metrics["amplitude_median"].abs() > 20))


good_clusters = metrics[keep_mask].index.values

# Load snippet
start_sec = 10       
duration_sec = 0.1    
start_sample = int(start_sec * fs)
n_samples = int(duration_sec * fs)

file_bytes = os.path.getsize(binary_file)
bytes_per_data_point = np.dtype(dtype).itemsize 
total_records = file_bytes // (total_channels * bytes_per_data_point)

memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_records, total_channels))
raw_segment = memmap_data[start_sample:start_sample + n_samples, :].T 

# --- FILTERING ---
# 1. Highpass for Spikes & Waveforms (300 Hz)
#b_hp, a_hp = butter(3, 300 / (fs / 2), btype='high')
b_hp, a_hp = butter(3,[500 / (fs / 2), 3000 / (fs / 2)],btype='bandpass')
highpass_segment = filtfilt(b_hp, a_hp, raw_segment, axis=1)

# 2. Highpass for Raw Data (10 Hz)
b_raw, a_raw = butter(3, 10 / (fs / 2), btype='high')
raw_segment = filtfilt(b_raw, a_raw, raw_segment, axis=1)

waveforms_dict = {}
waveform_samples = int(0.002 * fs) # 2 ms window

print('Calculating average waveforms for good neurons in selected channels...')
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
    snippets_filtered = filtfilt(b_hp, a_hp, snippets, axis=-1)
    mean_wf = np.mean(snippets_filtered, axis=0)
  
    row = cluster_info.loc[cluster_info['cluster_id'] == cluster]

    if row.empty:
        print(f"Warning: Unit {cluster} not found in cluster_info.tsv")
        continue

    phy_channel = int(row.iloc[0]['ch'])

    if phy_channel not in selected_channels or phy_channel in exclude_channels:
        continue

    waveforms_dict[cluster] = {
        'channel': phy_channel,
        'mean': mean_wf[phy_channel, :],
}

# --- PLOTTING ---
fig = plt.figure(figsize=(10, 12))

gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.4], wspace=0.15)

axes = {
    'raw': fig.add_subplot(gs[0]),
    'hp':  fig.add_subplot(gs[1]),
    'wf':  fig.add_subplot(gs[2])
}

t = np.linspace(0, duration_sec, n_samples)
wf_time_axis = np.linspace(0, 2.0, waveform_samples)

# --- PHYSICAL SPACING & GAIN CONSTANTS ---
spacing_raw = 800.0  
spacing_hp = 200.0
spacing_wf = 100.0  

# NEW: Multiplier to make raw signal wiggles visually taller
gain_raw = 1  #2.5

# Plot continuous signals
for row_idx, ch in enumerate(selected_channels):
    if ch in exclude_channels:
        continue  
        
    raw_uv = raw_segment[ch, :] * bit_to_uv
    hp_uv = highpass_segment[ch, :] * bit_to_uv
    
    # Apply gain_raw to the raw data signal
    axes['raw'].plot(t, (raw_uv * gain_raw) + (row_idx * spacing_raw), color='black', linewidth=0.6)
    axes['hp'].plot(t, hp_uv + (row_idx * spacing_hp), color='black', linewidth=0.6)

    axes['raw'].text(-0.02, row_idx * spacing_raw, f'Chn #{ch}', 
                     transform=axes['raw'].get_yaxis_transform(), 
                     ha='right', va='center', fontsize=9, color='sandybrown', style='italic')

# PLOT WAVEFORMS 
colors = plt.cm.tab20.colors 
channel_counts = {} 

for i, (cluster_id, data) in enumerate(waveforms_dict.items()):
    ch = data['channel']
    
    if ch not in selected_channels or ch in exclude_channels: 
        continue
        
    row_idx = selected_channels.index(ch)
    mean_uv = data['mean'] * bit_to_uv
    unit_color = colors[i % len(colors)]
    
    if ch not in channel_counts:
        channel_counts[ch] = 0
    
    count = channel_counts[ch]
    if count == 0: shift = 0.0      
    elif count == 1: shift = 0.35 
    elif count == 2: shift = -0.35 
    elif count == 3: shift = 0.80 
    else: shift = -0.70               
        
    channel_counts[ch] += 1
    
    final_offset_uv = (row_idx * spacing_wf) + (shift * spacing_wf)
    axes['wf'].plot(wf_time_axis, mean_uv + final_offset_uv, color=unit_color, linewidth=1.5)


# --- DRAW TRUE PHYSICAL SCALE BARS ---

# 1. Raw Data Scale Bar (10 µV / 10 ms)
y_start_raw = -(spacing_raw * 0.6) 
# Moved left (was 0.05, now 0.015)
x_start_raw = t[0]

# Height must be multiplied by gain_raw to remain strictly accurate
raw_bar_height = 20 * gain_raw
axes['raw'].plot([x_start_raw, x_start_raw], [y_start_raw, y_start_raw + raw_bar_height], color='black', lw=1.2, clip_on=False)
axes['raw'].text(x_start_raw - 0.005, y_start_raw + (raw_bar_height / 2), '20 µV', va='center', ha='right', fontsize=9)

axes['raw'].plot([x_start_raw, x_start_raw + 0.01], [y_start_raw, y_start_raw], color='black', lw=1.2, clip_on=False)
axes['raw'].text(x_start_raw + 0.005, y_start_raw - (spacing_raw * 0.08), '10 ms', va='top', ha='center', fontsize=9)


# 2. Highpass Data Scale Bar (10 µV / 10 ms)
y_start_hp = -(spacing_hp * 0.6)
# Moved left
x_start_hp = t[0]

axes['hp'].plot([x_start_hp, x_start_hp], [y_start_hp, y_start_hp + 20], color='black', lw=1.2, clip_on=False)
axes['hp'].text(x_start_hp - 0.005, y_start_hp + 5, '20 µV', va='center', ha='right', fontsize=9)

axes['hp'].plot([x_start_hp, x_start_hp + 0.01], [y_start_hp, y_start_hp], color='black', lw=1.2, clip_on=False)
axes['hp'].text(x_start_hp + 0.005, y_start_hp - (spacing_hp * 0.08), '10 ms', va='top', ha='center', fontsize=9)


# 3. Waveform Scale Bar (10 µV / 0.25 ms)
y_start_wf = -(spacing_wf * 0.6)
# Moved left (was 0.5, now 0.1)
wf_bar_x = -0.03

axes['wf'].plot([wf_bar_x, wf_bar_x], [y_start_wf, y_start_wf + 10], color='black', lw=1.2, clip_on=False)
axes['wf'].text(wf_bar_x - 0.1, y_start_wf + 5, '10 µV', va='center', ha='right', fontsize=9)

axes['wf'].plot([wf_bar_x, wf_bar_x + 0.25], [y_start_wf, y_start_wf], color='black', lw=1.2, clip_on=False)
axes['wf'].text(wf_bar_x + 0.125, y_start_wf - (spacing_wf * 0.15), '0.25 ms', va='top', ha='center', fontsize=9)


# --- FORMATTING, TITLES, & LIMITS ---

# Place Titles at the BOTTOM of the plots, centered under the traces
center_time = duration_sec / 2.0-0.03
center_wf = 0.98

axes['raw'].text(center_time, -(spacing_raw ), 'Raw', ha='center', va='top', fontsize=14)
axes['hp'].text(center_time, -(spacing_hp), 'Highpass', ha='center', va='top', fontsize=14)
axes['wf'].text(center_wf, -(spacing_wf), 'Waveforms', ha='center', va='top', fontsize=14)

# Lock limits so traces stay bounded and bottom titles fit cleanly
axes['raw'].set_ylim(-(spacing_raw * 1.4), n_selected * spacing_raw)
axes['hp'].set_ylim(-(spacing_hp * 1.4), n_selected * spacing_hp)
axes['wf'].set_ylim(-(spacing_wf * 1.4), n_selected * spacing_wf)
axes['raw'].set_xlim(t[0], t[-1])
axes['hp'].set_xlim(t[0], t[-1])

for name, ax in axes.items():
    ax.axis('off')  

# output
output_filename = 'kilosort_channels_66_to_76_updated.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Success! Saved clean layout to {output_filename}")

plt.show()
plt.close()