import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

binary_file = '20260804_data.bin'

fs = 30000
n_channels = 128
dtype = 'int16'
bit_to_uv = 0.25

# Phy channels you want to inspect
target_channels = [67, 69, 97, 109, 119]

# Data segment
start_sec = 10
duration_sec = 30

# Frequency bands

bands = [
    (0, 5),
    (5, 10),
    (10, 20),
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100)
]

# ==========================================
# LOAD BINARY
# ==========================================

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

# ==========================================
# FIGURE
# ==========================================

fig, axes = plt.subplots(
    len(target_channels),
    2,
    figsize=(12, 3 * len(target_channels))
)

for row, ch in enumerate(target_channels):

    trace = (
        data[start:end, ch].astype(float)
        * bit_to_uv
    )

    # Remove DC offset only
    trace = trace - np.mean(trace)

    # ======================================
    # 1. AMPLITUDE SPECTRUM
    # ======================================

    N = len(trace)

    fft_values = np.fft.rfft(trace)
    fft_freqs = np.fft.rfftfreq(N, d=1/fs)

    # Single-sided amplitude spectrum
    amplitude = 2.0 * np.abs(fft_values) / N

    mask = (fft_freqs >= 1) & (fft_freqs <= 5000)

    ax1 = axes[row, 0]

    ax1.plot(
        fft_freqs[mask],
        amplitude[mask],
        linewidth=1
    )

    ax1.axvline(
        500,
        linestyle='--',
        linewidth=1
    )

    ax1.axvline(
        3000,
        linestyle='--',
        linewidth=1
    )

    ax1.set_xlim(0, 5000)

    ax1.set_title(
        f'Phy Channel {ch} - Amplitude Spectrum'
    )

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (µV)')

    # ======================================
    # 2. BAND POWER DISTRIBUTION
    # ======================================

    freqs, psd = welch(
        trace,
        fs=fs,
        nperseg=8192
    )

    band_powers = []

    for low, high in bands:

        band_mask = (
            (freqs >= low) &
            (freqs < high)
        )

        power = np.trapz(
            psd[band_mask],
            freqs[band_mask]
        )

        band_powers.append(power)

    band_powers = np.array(band_powers)

    # Convert to percentage
    relative_power = (
        band_powers /
        np.sum(band_powers)
        * 100
    )

    labels = [
        f'{low}-{high}'
        for low, high in bands
    ]

    ax2 = axes[row, 1]

    bars = ax2.bar(
        labels,
        relative_power
    )

    ax2.set_title(
        f'Phy Channel {ch} - Frequency Distribution'
    )

    ax2.set_xlabel('Frequency Band (Hz)')
    ax2.set_ylabel('Relative Power (%)')

    ax2.tick_params(
        axis='x',
        rotation=45
    )

    # Put percentage above each bar
    for bar, value in zip(
        bars,
        relative_power
    ):
        ax2.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    'selected_channels_frequency_analysis.png',
    dpi=300,
    facecolor='white'
)

plt.show()