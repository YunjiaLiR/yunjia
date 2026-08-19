import spikeinterface.core as si
import spikeinterface.extractors as se
import spikeinterface.metrics as qm
import probeinterface as pi
import numpy as np
import os

# 1. Define your file paths
binary_file_path = r"F:/YST/yunjia/20260804_data.bin"
phy_dir = r"F:\YST\yunjia\20260804"

# 2. Load the curated Phy sorting results FIRST to get channel info
print("Loading Phy curation results...")
# Pass every label except 'good' into the exclude list
sorting = se.read_phy(phy_dir) #exclude_cluster_groups=["noise"]  exclude_cluster_groups=["noise", "mua", "unsorted"]

# 3. Build the probe map manually from your Kilosort files
print("Building and attaching probe layout...")
chan_pos = np.load(os.path.join(phy_dir, "channel_positions.npy"))
chan_map = np.load(os.path.join(phy_dir, "channel_map.npy")).flatten()

# Create a clean Probe object matching your exact physical positions
prb = pi.Probe(ndim=2)
prb.set_contacts(positions=chan_pos)
prb.set_device_channel_indices(np.arange(len(chan_pos)))

# 4. Load the raw binary recording using the correct channel count
print(f"Loading raw binary file (with {len(chan_pos)} channels mapped)...")
recording = se.read_binary(
    file_paths=binary_file_path, 
    sampling_frequency=30000, 
    num_channels=len(chan_pos), 
    dtype="int16",
    gain_to_uV=0.25
)

# Attach the probe layout to the recording
recording = recording.set_probe(prb, in_place=True)

# 5. Create a SortingAnalyzer
print("Creating Sorting Analyzer...")
analyzer = si.create_sorting_analyzer(
    sorting=sorting, 
    recording=recording, 
    format="memory"
)

# 6. Compute dependencies (ADDED "spike_amplitudes" here!)
analyzer.compute("random_spikes", method="uniform", max_spikes_per_unit=500)
analyzer.compute("waveforms", ms_before=1.0, ms_after=2.0)
analyzer.compute("templates", operators=["average", "median"])
analyzer.compute("spike_amplitudes") 

print("Computing quality metrics...")
metrics_list = ["isi_violation", "amplitude_cutoff", "presence_ratio", "amplitude_median"]
metrics = qm.compute_quality_metrics(analyzer, metric_names=metrics_list)

from spikeinterface.core import get_template_extremum_amplitude

template_amps = get_template_extremum_amplitude(
    analyzer,
    peak_sign="neg",
    abs_value=True
)

for unit_id in [77, 168, 180]:
    print(
        f"Unit {unit_id} | "
        f"amplitude_median = {abs(metrics.loc[unit_id, 'amplitude_median']):.2f} uV | "
        f"template main-channel amplitude = {template_amps[unit_id]:.2f} uV"
    )

# Save all metrics to a CSV file
metrics.to_csv(r"F:\YST\yunjia\20260804\all_quality_metrics.csv")
print("Metrics saved to F:\\YST\\yunjia\\20260804\\all_quality_metrics.csv")

keep_mask = (
    (metrics["isi_violations_ratio"] < 0.5) &
    (metrics["amplitude_cutoff"] < 0.1) &
    (metrics["presence_ratio"] > 0.8) & (metrics["amplitude_median"].abs() > 15)
)
high_quality_unit_ids = metrics[keep_mask].index.values
final_sorting = sorting.select_units(high_quality_unit_ids)

print("\n=== RESULTS ===")
print(f"Total units from Phy: {len(sorting.unit_ids)}")
print(f"Units passing objective post-curation: {len(final_sorting.unit_ids)}")