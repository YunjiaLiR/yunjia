import os

# file_path = 'F:/YST/yunjia/20260703_raw data_20260316/20260316_data.bin'
file_path = 'F:/YST/yunjia/20260210_data.bin'
num_channels = 128   
sampling_rate = 30000
bytes_per_sample = 2   # 2 bytes for int16 (standard for Kilosort)

# 2. Get the file size in bytes
file_size_bytes = os.path.getsize(file_path)

# 3. Calculate total duration
total_samples_per_channel = file_size_bytes / (num_channels * bytes_per_sample)
total_time_seconds = total_samples_per_channel / sampling_rate
total_time_minutes = total_time_seconds / 60

print(f"File Path: {file_path} Total Recording Time: {total_time_seconds:.2f} seconds ({total_time_minutes:.2f} minutes)")