import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

binary_file = '20260804_data.bin'

fs = 30000
n_channels = 128
dtype = 'int16'
bit_to_uv = 0.25

target_channels = [67,69,97,109,119]


start_sec = 10
duration_sec = 30

file_bytes = os.path.getsize(binary_file)
total_samples = file_bytes // (
    n_channels * np.dtype(dtype).itemsize
)

data = np.memmap(
    binary_file,
    dtype=dtype,
    mode='r',
    shape=(total_samples, n_channels)
)

start = int(start_sec * fs)
end = int((start_sec + duration_sec) * fs)

fig, axes = plt.subplots(
    len(target_channels),
    1,
    figsize=(7, 3 * len(target_channels)),
    sharex=True
)

if len(target_channels) == 1:
    axes = [axes]

for ax, ch in zip(axes, target_channels):

    trace = (
        data[start:end, ch].astype(float)
        * bit_to_uv
    )

    # Remove DC offset
    trace = trace - np.mean(trace)

    # Welch PSD
    freqs, psd = welch(
        trace,
        fs=fs,
        nperseg=8192
    )

    # show 1–5000 Hz
    mask = (freqs >= 1) & (freqs <= 5000)

    ax.semilogy(
        freqs[mask],
        psd[mask],
        linewidth=1
    )

    # mentor's proposed spike band
    ax.axvline(
        500,
        linestyle='--',
        linewidth=1
    )

    ax.axvline(
        3000,
        linestyle='--',
        linewidth=1
    )

    ax.set_title(f'Phy Channel {ch}')
    ax.set_ylabel('PSD (µV²/Hz)')

axes[-1].set_xlabel('Frequency (Hz)')

plt.tight_layout()
plt.savefig(
    'selected_channels_frequency_spectrum.png',
    dpi=300,
    facecolor='white'
)

plt.show()