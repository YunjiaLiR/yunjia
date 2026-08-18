import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. Define Your Data Parameters ---
file_path = 'F:/YST/yunjia/20260210_data.bin' 
# file_path = 'F:/YST/yunjia/20260210_kilosort_results/temp_wh.dat' 
num_channels = 112                  
sampling_rate = 30000            
channel = 0          


data = np.memmap(file_path, dtype='int16', mode='r')

# Reshape the data to (number of samples, number of channels)
num_samples = len(data) // num_channels
data = data.reshape((num_samples, num_channels))

# Extract exactly 10 seconds of data from the target channel
# (30000 Hz * 10 seconds = 300,000 samples)
chunk_size = sampling_rate * 10 
# Grab a chunk from the middle of the recording to avoid startup artifacts
start_idx = num_samples // 2 
end_idx = start_idx + chunk_size

# Get the raw voltage trace
voltage_trace = data[start_idx:end_idx, channel]

# --- 3. Calculate the PSD using Welch's Method ---
# nperseg controls the frequency resolution. A 1-second window is great.
frequencies, psd = signal.welch(voltage_trace, fs=sampling_rate, nperseg=sampling_rate)

# --- 4. Plot the Results ---
plt.figure(figsize=(10, 6))

# We usually plot PSD on a logarithmic scale for the y-axis
plt.semilogy(frequencies, psd, color='blue')

# Limit the x-axis to the biological/noise range (0 to 5000 Hz)
plt.xlim(0, 5000) 

plt.title(f'Power Spectral Density (Channel {channel})')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power (V^2/Hz)')
plt.grid(True, which="both", ls="--", alpha=0.5)

# Highlight typical 50/60Hz mains noise and your 300Hz filter mark
plt.axvline(x=60, color='red', linestyle=':', label='60 Hz Mains') # Change to 50 if in EU/UK/Asia
plt.axvline(x=300, color='green', linestyle=':', label='300 Hz Filter Edge')
plt.legend()
plt.savefig(f"Channel {channel} PSD (channel starts from 0)", dpi=300)
plt.show()
