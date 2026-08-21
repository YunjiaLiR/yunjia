# Multichannel Neural Recording Analysis

Python utilities developed for preprocessing, quality control, spike-sorting review, and post-curation analysis of high-channel-count extracellular electrophysiology recordings.

The repository focuses on the analysis stage of an implanted-electrode workflow: validating raw recordings, checking probe geometry, working with Kilosort/Phy outputs, applying objective post-curation quality criteria, and generating unit- and array-level electrophysiology figures.

> **Repository scope:** analysis code only. Raw recordings, Kilosort/Phy session outputs, generated figures, and experiment-derived result tables are intentionally excluded.

## Analysis workflow

```text
Raw multichannel recording
        |
        v
1. Recording integrity / signal inspection
   - recording duration
   - raw voltage traces
   - PSD / electrical-noise inspection
   - probe / shank geometry
        |
        v
2. Spike sorting
   - Kilosort
   - experiment-specific channel mapping
        |
        v
3. Manual curation
   - Phy
   - inspect / split / merge candidate units
        |
        v
4. Objective post-curation quality control
   - ISI violations
   - amplitude cutoff
   - presence ratio
   - waveform amplitude
        |
        v
5. Electrophysiology analysis and visualization
   - waveform morphology
   - firing rate
   - raster plots
   - SNR
   - autocorrelogram rise time
   - amplitude
   - peak-to-trough metrics
   - spectrum / raw vs high-pass signal
   - spatial firing-rate map
```

## Repository structure

```text
.
├── utilities/
│   ├── raw_data_length.py      # Estimate recording duration from binary file size
│   ├── plot_raw_data.py        # Inspect raw channel voltage traces
│   ├── psd.py                  # Power spectral density / noise inspection
│   ├── spike_length.py         # Compare spike-output duration with raw recording
│   └── shank.py                # Visualize Kilosort probe geometry / shank mapping
│
├── curation/
│   └── curation.py             # SpikeInterface-based objective post-curation metrics
│
├── plot_figures/
│   ├── acg_rise_time.py
│   ├── amplitude.py
│   ├── firing_rate.py
│   ├── peak_trough_ratio.py
│   ├── raster.py
│   ├── raw_hp_wave.py
│   ├── raw_hp_wave_short.py
│   ├── snr.py
│   ├── spectrum.py
│   ├── topomap.py
│   ├── trough_to_peak_duration.py
│   └── waveform.py
│
├── requirements.txt
└── .gitignore
```

## Analysis SOP

A more detailed description of the preprocessing, spike-sorting, manual-curation, and quantitative quality-control workflow is provided in:

- `docs/spike_sorting_sop.tex`

The SOP documents the processing order, channel-exclusion criteria, Kilosort parameters, manual Phy curation, and quantitative SUA selection criteria used in this repository. It is intentionally written without raw data, session-specific results, or experimental figures so that it can be adapted to other recordings.

## Signal-quality features

The analysis scripts cover several complementary aspects of extracellular recording quality.

### Recording-level checks

- **Raw voltage inspection** for clipping, disconnection, large artifacts, and abnormal channel behavior.
- **PSD analysis** for frequency-specific electrical contamination.
- **Recording/spike-output duration checks** for detecting incomplete processing or data discontinuity.
- **Probe geometry checks** using Kilosort channel-map and position files.

### Unit-level metrics

The current post-curation filtering script uses:

| Metric | Current criterion | Purpose |
|---|---:|---|
| ISI violations ratio | `< 0.5` | Reject units with excessive refractory-period violations |
| Amplitude cutoff | `< 0.1` | Limit estimated missing spikes caused by amplitude truncation |
| Presence ratio | `> 0.8` | Require temporal presence across most of the recording |
| Median amplitude | `abs(amplitude_median) > 20 µV` | Remove very-low-amplitude units |

These thresholds are analysis choices in the current workflow, not universal standards. They should be justified or re-tuned for other recording systems and experiments.

### Waveform / physiology summaries

The plotting scripts additionally inspect features such as:

- waveform amplitude;
- trough-to-peak duration;
- peak-to-trough ratio;
- firing rate;
- autocorrelogram rise time;
- signal-to-noise ratio;
- spatial distribution of firing rate across the electrode geometry.

## Data assumptions

The scripts were developed around extracellular recordings with typical parameters such as:

```text
sampling rate: 30,000 Hz
binary dtype: int16
channel count: experiment dependent (currently commonly 128)
voltage conversion: 0.25 µV / bit in the current Blackrock setup
```

Do not assume these parameters are correct for another acquisition system.

Kilosort/Phy-derived scripts expect some combination of files such as:

```text
spike_times.npy
spike_clusters.npy
channel_map.npy
channel_positions.npy
cluster_info.tsv
all_quality_metrics.csv
```

These files are **not distributed in this repository**.

## Installation

Create an isolated Python environment, then install:

```bash
pip install -r requirements.txt
```

Core Python dependencies are NumPy, pandas, SciPy, Matplotlib, SpikeInterface, and ProbeInterface.

Kilosort and Phy are separate tools used upstream for spike sorting and manual curation.

## Example: recording duration

The refined utility accepts recording parameters from the command line:

```bash
python utilities/raw_data_length.py path/to/recording.bin \
    --channels 128 \
    --sampling-rate 30000 \
    --dtype int16
```

## Example: inspect raw signal

```bash
python utilities/plot_raw_data.py path/to/recording.bin \
    --channels 128 \
    --sampling-rate 30000 \
    --channel 10 \
    --start 0 \
    --duration 5 \
    --bit-to-uv 0.25
```

## Example: probe geometry

```bash
python utilities/shank.py path/to/kilosort_or_phy_folder
```

The folder should contain `channel_positions.npy` and `channel_map.npy`. If `channel_shanks.npy` is unavailable, the script falls back to grouping channels by X-coordinate for visualization.

## Example: objective post-curation metrics

```bash
python curation/curation.py \
    --binary path/to/recording.bin \
    --phy path/to/phy_folder \
    --sampling-rate 30000 \
    --gain-to-uv 0.25 \
    --output path/to/all_quality_metrics.csv
```

The script reads the curated Phy sorting, attaches channel geometry, computes waveform/template-dependent quality metrics with SpikeInterface, and reports the units passing the current objective criteria.

## Current limitations

This repository represents an active research-analysis workflow rather than a finished software package.

- Several scripts in `plot_figures/` still use experiment-specific configuration blocks such as session folder names.
- Channel count, voltage conversion, excluded/reference channels, and binary-file naming conventions must match the acquisition setup.
- `topomap.py` requires the raw binary file and Kilosort/Phy output folder to correspond to the same recording.
- Quality-control thresholds should not be transferred to a new dataset without validation.
- Raw data and derived experiment outputs are intentionally excluded.

A future refactor can centralize session configuration and expose consistent command-line interfaces across all figure scripts.

## Research context

This code was developed as part of an implanted-electrode neural-recording analysis workflow involving:

1. multichannel extracellular acquisition;
2. signal-quality inspection;
3. Kilosort spike sorting;
4. manual Phy curation;
5. objective unit-quality filtering;
6. waveform, firing-rate, temporal, noise, and spatial analysis.

The broader research pipeline may include experimental design, receptive-field mapping, anatomical verification, and downstream neural decoding. Those stages are outside the current scope of this repository unless corresponding code is added later.

## Data and privacy

No raw neural recordings should be committed to this repository.

The `.gitignore` excludes common electrophysiology data, Kilosort/Phy outputs, generated figures, documents, archives, environments, and local IDE files.

## License / reuse

No open-source license is included at this stage. Before publicly distributing or licensing internship/research code, confirm the applicable laboratory, institution, or employer IP and publication policy.
