import numpy as np

spike_times = np.load('F:/YST/yunjia/20260703_raw data_20260316/kilosort_results/spike_times.npy')

print(f"Total spikes detected: {len(spike_times)}")

# 3. View the first 10 spike times (these are in SAMPLES, not seconds)
print(f"First 10 spikes (samples): {spike_times[:10].flatten()}")

sampling_rate = 30000 

# Get the very last spike in the array
last_spike_sample = np.max(spike_times) 

# Convert that sample number into seconds
last_spike_seconds = last_spike_sample / sampling_rate 

print(f"\nThe very last spike was detected at: {last_spike_seconds:.2f} seconds")