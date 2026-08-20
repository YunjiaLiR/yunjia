import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt

binary_file = '20260804_data.bin'

fs = 30000
n_channels = 128
dtype = 'int16'
bit_to_uv = 0.25

# Phy channels to inspect
target_channels = [67, 69, 97, 109, 119]

# Data segment
start_sec = 10
duration_sec = 30

# Frequency bands WITHIN the 500–3000 Hz spike band
bands = [
    (500, 750),
    (750, 1000),
    (1000, 1250),
    (1250, 1500),
    (1500, 1750),
    (1750, 2000),
    (2000, 2250),
    (2250, 2500),
    (2500, 2750),
    (2750, 3000)
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
# 500–3000 Hz BAND-PASS FILTER
# ==========================================

b_bp, a_bp = butter(
    3,
    [500 / (fs / 2), 3000 / (fs / 2)],
    btype='bandpass'
)

# ==========================================
# FIGURE
# ==========================================

fig, axes = plt.subplots(
    len(target_channels),
    2,
    figsize=(12, 3 * len(target_channels))
)

for row, ch in enumerate(target_channels):

    # Raw recording in µV
    trace = (
        data[start:end, ch].astype(float)
        * bit_to_uv
    )

    trace = trace - np.mean(trace)

    # Apply 500–3000 Hz BP
    trace_bp = filtfilt(
        b_bp,
        a_bp,
        trace
    )

    # ======================================
    # 1. AMPLITUDE SPECTRUM AFTER BP
    # ======================================

    N = len(trace_bp)

    fft_values = np.fft.rfft(trace_bp)

    fft_freqs = np.fft.rfftfreq(
        N,
        d=1/fs
    )

    amplitude = (
        2.0 * np.abs(fft_values) / N
    )

    # Only show the actual pass band
    mask = (
        (fft_freqs >= 500) &
        (fft_freqs <= 3000)
    )

    ax1 = axes[row, 0]

    ax1.plot(
        fft_freqs[mask],
        amplitude[mask],
        linewidth=1
    )

    ax1.set_xlim(500, 3000)

    ax1.set_title(
        f'Phy Channel {ch} - 500–3000 Hz Amplitude Spectrum'
    )

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Amplitude (µV)')

    # ======================================
    # 2. FREQUENCY DISTRIBUTION AFTER BP
    # ======================================

    freqs, psd = welch(
        trace_bp,
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

    band_powers = np.array(
        band_powers
    )

    # Percentage of power WITHIN 500–3000 Hz
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
        f'Phy Channel {ch} - 500–3000 Hz Power Distribution'
    )

    ax2.set_xlabel(
        'Frequency Band (Hz)'
    )

    ax2.set_ylabel(
        'Relative Power (%)'
    )

    ax2.tick_params(
        axis='x',
        rotation=45
    )

    # Percentage above each bar
    for bar, value in zip(
        bars,
        relative_power
    ):
        ax2.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height(),
            f'{value:.1f}%',
            ha='center',
            va='bottom',
            fontsize=8
        )

plt.tight_layout()

plt.savefig(
    'selected_channels_500_3000Hz_frequency_analysis.png',
    dpi=300,
    facecolor='white'
)

plt.show()