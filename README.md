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
│   ├── raw_data_length.py
│   ├── plot_raw_data.py
│   ├── psd.py
│   ├── spike_length.py
│   └── shank.py
│
├── curation/
│   └── curation.py
│
├── analysis/
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
├── SOP.tex
├── requirements.txt
└── .gitignore
```

## Analysis SOP

A more detailed description of the preprocessing, spike-sorting, manual-curation, and quantitative quality-control workflow is provided in:

* `SOP.tex`

The SOP documents the processing order, channel-exclusion criteria, Kilosort parameters, manual Phy curation, and quantitative SUA selection criteria used in this repository.

It is intentionally written without raw data, session-specific results, or experimental figures so that the workflow can be adapted to other recordings.

## Signal-quality features

The analysis scripts cover several complementary aspects of extracellular recording quality.

### Recording-level checks

* **Raw voltage inspection** for clipping, disconnection, large artifacts, and abnormal channel behaviour.
* **PSD analysis** for frequency-specific electrical contamination.
* **Recording/spike-output duration checks** for detecting incomplete processing or data discontinuity.
* **Probe geometry checks** using Kilosort channel-map and position files.

### Unit-level quality metrics

The current post-curation filtering workflow uses:

| Metric               |               Current criterion | Purpose                                                       |
| -------------------- | ------------------------------: | ------------------------------------------------------------- |
| ISI violations ratio |                         `< 0.5` | Reject units with excessive refractory-period violations      |
| Amplitude cutoff     |                         `< 0.1` | Limit estimated missing spikes caused by amplitude truncation |
| Presence ratio       |                         `> 0.8` | Require temporal presence across most of the recording        |
| Median amplitude     | `abs(amplitude_median) > 20 µV` | Remove very-low-amplitude units                               |

These thresholds are analysis choices used in the current workflow rather than universal standards. They should be validated or adjusted before being applied to another recording system or experimental dataset.

### Waveform and physiology analysis

The analysis scripts additionally examine features including:

* waveform amplitude;
* trough-to-peak duration;
* peak-to-trough ratio;
* firing rate;
* raster activity;
* autocorrelogram rise time;
* signal-to-noise ratio;
* spectral characteristics;
* spatial distribution of firing rate across the electrode geometry.

## Data assumptions

The scripts were developed around extracellular recordings with typical parameters such as:

```text
sampling rate: 30,000 Hz
binary dtype: int16
channel count: experiment dependent, commonly 128
voltage conversion: 0.25 µV / bit for the current Blackrock setup
```

These parameters should not be assumed to be correct for another acquisition system.

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

Create an isolated Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

Core dependencies include:

* NumPy
* pandas
* SciPy
* Matplotlib
* SpikeInterface
* ProbeInterface

Kilosort and Phy are separate tools used upstream for spike sorting and manual curation.

## Example: recording duration

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

The folder should contain:

```text
channel_positions.npy
channel_map.npy
```

If `channel_shanks.npy` is unavailable, the script falls back to grouping channels by X-coordinate for visualization.

## Objective post-curation quality control

The current `curation/curation.py` script reads curated Phy output, attaches the probe geometry, computes waveform-dependent quality metrics using SpikeInterface, and identifies units passing the current objective criteria.

Before running the script, update the recording and Phy folder paths near the beginning of the file:

```python
binary_file_path = r"path/to/recording.bin"
phy_dir = r"path/to/phy_folder"
```

Then run:

```bash
python curation/curation.py
```

The script computes:

```text
ISI violations ratio
amplitude cutoff
presence ratio
median amplitude
```

and saves the resulting quality-metric table as `all_quality_metrics.csv`.

The current SUA selection criteria are:

```text
ISI violations ratio < 0.5
amplitude cutoff < 0.1
presence ratio > 0.8
|median amplitude| > 20 µV
```

## Current limitations

This repository represents an active research-analysis workflow rather than a finished software package.

* Several scripts in `analysis/` use experiment-specific configuration blocks such as recording names and session folders.
* Channel count, voltage conversion, excluded/reference channels, and binary-file naming conventions must match the acquisition setup.
* `topomap.py` requires the raw binary recording and Kilosort/Phy output folder to correspond to the same recording.
* Quality-control thresholds should not be transferred to a new dataset without validation.
* Raw data and derived experimental outputs are intentionally excluded.

The scripts are kept close to the versions used during analysis to preserve the original processing workflow.

## Research context

This code was developed as part of an implanted-electrode neural-recording analysis workflow involving:

1. multichannel extracellular acquisition;
2. signal-quality inspection;
3. Kilosort spike sorting;
4. manual Phy curation;
5. objective unit-quality filtering;
6. waveform, firing-rate, temporal, noise, spectral, and spatial analysis.

The broader research pipeline may include experimental design, receptive-field mapping, anatomical verification, and downstream neural decoding. Those stages are outside the current scope of this repository unless corresponding code is added later.

## Data and privacy

No raw neural recordings should be committed to this repository.

The `.gitignore` excludes common electrophysiology data, Kilosort/Phy outputs, generated figures, documents, archives, environments, and local system files.

## License / reuse

No open-source license is included at this stage.

Before publicly distributing or licensing research or internship code, the applicable laboratory, institution, or employer intellectual-property and publication policies should be confirmed.
