import numpy as np
import matplotlib.pyplot as plt

# --- Your Settings ---
file_path = 'F:/YST/yunjia/20260210_data.bin'
num_channels = 128    # CHANGE THIS to your actual channel count
sampling_rate = 30000  # CHANGE THIS to your actual sampling rate
start_time_sec = 0  # Start looking 1 second before the crash
duration_sec = 300    # Load 2 seconds of data
bit_volts = 0.25

# --- The Math ---
samples_to_read = int(duration_sec * sampling_rate * num_channels)
offset_bytes = int(start_time_sec * sampling_rate * num_channels * 2) # 2 bytes per int16

# --- Read and Plot ---
# Load just this tiny 2-second chunk to save memory
data = np.fromfile(file_path, dtype=np.int16, count=samples_to_read, offset=offset_bytes)
data = data.reshape(-1, num_channels)
data_uV = data.astype(np.float32) * bit_volts

channel =10
plt.figure(figsize=(12, 4))
plt.plot(data_uV[:, channel]) #
plt.title(f"Channel {channel} Raw Data at {start_time_sec} - {start_time_sec+duration_sec} seconds")
plt.xlabel("Samples")
plt.ylabel("Voltage (uV)")
# plt.ylim(-50,50)
plt.savefig(f"Channel {channel} Raw Data at {start_time_sec} - {start_time_sec+duration_sec} seconds", dpi=300)
plt.show()