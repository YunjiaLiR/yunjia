import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import scipy.signal as signal
from scipy.signal import butter, filtfilt

binary_file = '20260302_data.bin'
folder_path = '20260302'
fs = 30000
n_channels = 128
dtype = 'int16'
bit_to_uv = 0.25

# --- MULTIPLE TARGET CLUSTERS & COLORS ---
target_clusters = [68,70,71]    # Array of target cluster IDs
cluster_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

target_channel = 35       # Peak channel
start_sec = 1.2          # Start time in seconds
duration = 0.5            # Window duration in seconds


print("Loading data...")
spike_times = np.load(os.path.join(folder_path, 'spike_times.npy')).flatten()
spike_clusters = np.load(os.path.join(folder_path, 'spike_clusters.npy')).flatten()

# Load memory map
file_bytes = os.path.getsize(binary_file)
total_samples = file_bytes // (n_channels * np.dtype(dtype).itemsize)
memmap_data = np.memmap(binary_file, dtype=dtype, mode='r', shape=(total_samples, n_channels))

# --- Extract Continuous Traces ---
start_sample = int(start_sec * fs)
end_sample = int((start_sec + duration) * fs)

raw_data = memmap_data[start_sample:end_sample, target_channel] * bit_to_uv

b, a = butter(3, 10 / (fs / 2), btype='high')
raw_data = filtfilt(b, a, raw_data)

# Highpass filter
# b, a = butter(3, 300 / (fs / 2), btype='high')
b, a = butter(3,[500 / (fs / 2), 3000 / (fs / 2)],btype='bandpass')
hp_data = filtfilt(b, a, raw_data)

t = np.linspace(0, duration, len(raw_data))

# --- Extract Spikes and Waveforms for ALL Target Clusters ---
pre_samples = int(0.001 * fs)   # 1 ms
post_samples = int(0.002 * fs)  # 2 ms
wf_t = np.linspace(-1, 2, pre_samples + post_samples)

cluster_data = {}
all_spikes_in_window = []

for idx, cluster_id in enumerate(target_clusters):
    color = cluster_colors[idx % len(cluster_colors)]
    
    # Get spikes for this cluster
    unit_spikes = spike_times[spike_clusters == cluster_id]
    unit_spikes_sec = (unit_spikes - start_sample) / fs
    spikes_in_window = unit_spikes_sec[(unit_spikes_sec >= 0) & (unit_spikes_sec <= duration)]
    
    all_spikes_in_window.extend(spikes_in_window)

    # Calculate Mean Waveform
    snippets = []
    spikes_to_use = np.random.choice(unit_spikes, min(300, len(unit_spikes)), replace=False)
    for spk in spikes_to_use:
        if spk > pre_samples and spk < total_samples - post_samples:
            snippet = memmap_data[spk - pre_samples : spk + post_samples, target_channel] * bit_to_uv
            snippets.append(snippet)

    if len(snippets) > 0:
        snippets = np.array(snippets)
        snippets_filtered = filtfilt(b, a, snippets, axis=1)
        unit_wf = np.mean(snippets_filtered, axis=0)
        unit_wf_std = np.std(snippets_filtered, axis=0)

        # Baseline subtraction
        baseline = np.mean(unit_wf[:int(0.5 * fs / 1000)])
        unit_wf = unit_wf - baseline
    else:
        unit_wf = np.zeros(pre_samples + post_samples)
        unit_wf_std = np.zeros(pre_samples + post_samples)

    # Store for plotting
    cluster_data[cluster_id] = {
        'color': color,
        'spikes_in_window': spikes_in_window,
        'wf_mean': unit_wf,
        'wf_std': unit_wf_std
    }

print("Plotting...")
fig = plt.figure(figsize=(12, 8))
gs = gridspec.GridSpec(4, 6, height_ratios=[1, 1.5, 1, 1.8], wspace=0.3, hspace=0.4)

ax_raw  = fig.add_subplot(gs[0, 0:5])
ax_spec = fig.add_subplot(gs[1, 0:5])
ax_hp   = fig.add_subplot(gs[2, 0:5])
ax_zoom = fig.add_subplot(gs[3, 0:5])
ax_unit = fig.add_subplot(gs[3, 5]) 

# --- ROW 1: RAW DATA ---
ax_raw.plot(t, raw_data, color='black', linewidth=0.8)
ax_raw.axis('off')
ax_raw.text(0, 1.1, f'Chn #{target_channel} (Raw)', transform=ax_raw.transAxes, fontsize=12, va='bottom')

# Moved 'Raw' label to the top-left margin
ax_raw.text(-0.05, 1.1, 'Raw', transform=ax_raw.transAxes, color='sandybrown', fontsize=12, style='italic', va='bottom')

# Lock x-limits so the raw trace size stays fixed
ax_raw.set_xlim(t[0], t[-1])

# --- Scale Bar Setup (Smaller: 100 µV / 50 ms, Bottom-Left) ---
ptp_raw = np.ptp(raw_data)
y_start_raw = np.min(raw_data) - (ptp_raw * 0.04)  # Shifted slightly down below trace
x_start_raw = t[0] - 0.015                         # Shifted slightly left into margin

# 1. Vertical Line (100 µV)
ax_raw.plot([x_start_raw, x_start_raw], [y_start_raw, y_start_raw + 100], color='black', lw=1.2, clip_on=False)
ax_raw.text(x_start_raw - 0.008, y_start_raw + 50, '100 µV', va='center', ha='right', fontsize=8)

# 2. Horizontal Line (50 ms = 0.05 s)
ax_raw.plot([x_start_raw, x_start_raw + 0.05], [y_start_raw, y_start_raw], color='black', lw=1.2, clip_on=False)
ax_raw.text(x_start_raw + 0.025, y_start_raw - (ptp_raw * 0.05), '50 ms', va='top', ha='center', fontsize=8)


# --- ROW 2: SPECTROGRAM ---
f_spec, t_spec, Sxx = signal.spectrogram(raw_data, fs, nperseg=1024, noverlap=512)
freq_mask = f_spec <= 250
img = ax_spec.pcolormesh(t_spec, f_spec[freq_mask], 10 * np.log10(Sxx[freq_mask, :]), shading='gouraud', cmap='turbo')
ax_spec.spines['top'].set_visible(False)
ax_spec.spines['right'].set_visible(False)
ax_spec.spines['bottom'].set_visible(False)
ax_spec.set_xticks([]) 
ax_spec.set_ylabel('Frequency\n(Hz)', fontsize=11)
ax_spec.text(-0.05, 1.1, 'Lowpass', transform=ax_spec.transAxes, color='sandybrown', fontsize=12, style='italic', va='bottom')
cbar = fig.colorbar(img, ax=ax_spec, fraction=0.02, pad=0.01)
cbar.set_label('Power\n(dB)', rotation=270, labelpad=20)

# --- ROW 3: HIGHPASS DATA ---

ax_hp.plot(t, hp_data, color='black', linewidth=0.5)
ax_hp.axis('off')
ax_hp.text(-0.05, 1.1, 'Highpass', transform=ax_hp.transAxes, color='sandybrown', fontsize=12, style='italic', va='bottom')

# Lock x-limits so the highpass trace size stays fixed
ax_hp.set_xlim(t[0], t[-1])

# --- Scale Bar Setup (25 µV / 20 ms, Shifted Left & Down) ---
ptp_val = np.ptp(hp_data)
y_start = np.min(hp_data) - (ptp_val * 0.2)  # Shifted down slightly below the lowest trace
x_start = t[0] - 0.008                         # Shifted slightly left into the margin

# 1. Vertical Line (25 µV)
ax_hp.plot([x_start, x_start], [y_start, y_start + 25], color='black', lw=1.2, clip_on=False)
ax_hp.text(x_start - 0.008, y_start + 12.5, '25 µV', va='center', ha='right', fontsize=8)

# 2. Horizontal Line (20 ms = 0.02 s)
ax_hp.plot([x_start, x_start + 0.02], [y_start, y_start], color='black', lw=1.2, clip_on=False)
ax_hp.text(x_start + 0.01, y_start - (ptp_val * 0.05), '20 ms', va='top', ha='center', fontsize=8)

# Zoom window centering logic across all spikes in window
if len(all_spikes_in_window) > 0:
    zoom_center = all_spikes_in_window[0]
else:
    zoom_center = 10
zoom_start = max(0, zoom_center - 0.1)
zoom_end = min(duration, zoom_start + 0.2)

rect_height = ptp_val
rect = patches.Rectangle((zoom_start, np.min(hp_data)), zoom_end - zoom_start, rect_height, 
                         linewidth=1, edgecolor='steelblue', facecolor='none', alpha=0.5)
ax_hp.add_patch(rect)

# --- ROW 4: ZOOMED HIGHPASS ---
zoom_mask = (t >= zoom_start) & (t <= zoom_end)
ax_zoom.plot(t[zoom_mask], hp_data[zoom_mask], color='black', linewidth=0.8)
for spine in ax_zoom.spines.values():
    spine.set_edgecolor('steelblue')
    spine.set_alpha(0.5)
ax_zoom.set_xticks([])
ax_zoom.set_yticks([])

# Add cluster-colored arrows pointing to spikes in the zoom window
y_arrow = np.min(hp_data[zoom_mask]) - 20
for cluster_id, info in cluster_data.items():
    c_color = info['color']
    for st in info['spikes_in_window']:
        if zoom_start <= st <= zoom_end:
            ax_zoom.annotate('', xy=(st, y_arrow + 30), xytext=(st, y_arrow - 20),
                             arrowprops=dict(arrowstyle='->', color=c_color, alpha=0.8, lw=1.5))

# Zoom Scale bar
zx, zy = zoom_start, y_arrow - 20
ax_zoom.plot([zx, zx + 0.005], [zy, zy], color='black', lw=2, clip_on=False) 
ax_zoom.plot([zx, zx], [zy, zy + 30], color='black', lw=2, clip_on=False)
ax_zoom.text(zx + 0.0025, zy - 10, '5 ms', ha='center', va='top', fontsize=10)
ax_zoom.text(zx - 0.005, zy + 15, '30 µV', ha='right', va='center', fontsize=10, rotation=90)

# --- ROW 4 (RIGHT): ISOLATED UNITS OVERLAID ---
# Set vertical gap between stacked units (in µV)
offset_step = 70  # Increase this if waveforms still overlap, or decrease for a tighter stack

min_uys = []
for idx, (cluster_id, info) in enumerate(cluster_data.items()):
    c_color = info['color']
    wf_mean = info['wf_mean']
    wf_std = info['wf_std']
    
    # Calculate vertical offset (Unit 0 sits at 0 µV, Unit 1 shifts down by -60 µV, etc.)
    offset = -idx * offset_step
    shifted_mean = wf_mean + offset
    
    # Plot standard deviation shadow with offset
    ax_unit.fill_between(wf_t, shifted_mean - wf_std, shifted_mean + wf_std, color=c_color, alpha=0.15)
    
    # Plot mean waveform line with offset
    ax_unit.plot(wf_t, shifted_mean, color=c_color, linewidth=1.5, label=f'Unit #{cluster_id}')
    
    # Plot a baseline reference line for each individual stacked unit
    ax_unit.axhline(offset, color='gray', linestyle='--', linewidth=0.5, alpha=0.4)
    
    min_uys.append(np.min(shifted_mean - wf_std))

ax_unit.axis('off')
# Place the legend outside to the right of ax_unit
ax_unit.legend(
    frameon=False, 
    fontsize=9, 
    loc='upper left',            # Anchor point of the legend box
    bbox_to_anchor=(1.02, 1.0)   # x > 1.0 pushes it outside to the right
)

# Unit Scale Bar
# Unit Scale Bar
uy_min = min(min_uys) if len(min_uys) > 0 else -50

# --- 1. Define vertical padding to push the scale bar down ---
padding = 20  # Increase this number (e.g., 30 or 40) to move it even lower
y_bar = uy_min - padding

# --- 2. Draw lines using y_bar as the new base ---
# Horizontal line (0.5 ms)
ax_unit.plot([-1.0, -0.5], [y_bar, y_bar], color='black', lw=2, clip_on=False) 

# Vertical line (25 µV)
ax_unit.plot([-1.0, -1.0], [y_bar, y_bar + 25], color='black', lw=2, clip_on=False)

# --- 3. Draw text labels relative to y_bar ---
ax_unit.text(-0.75, y_bar - 10, '0.5 ms', ha='center', va='top', fontsize=9)
ax_unit.text(-0.8, y_bar + 12.5, '25 µV', ha='left', va='center', fontsize=9)

plt.tight_layout()
output_filename = f'units_{"_".join(map(str, target_clusters))}_summary.png'
plt.savefig(output_filename, dpi=300, facecolor='white', bbox_inches='tight')
print(f"Success! Saved to {output_filename}")
plt.show()