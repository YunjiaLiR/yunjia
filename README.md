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