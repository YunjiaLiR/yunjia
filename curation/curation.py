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

templates_ext = analyzer.get_extension("templates")
average_templates = templates_ext.get_data(operator="average")

unit_ids = analyzer.unit_ids

for unit_id in [77, 168, 180]:

    if unit_id not in unit_ids:
        print(f"Unit {unit_id} not found")
        continue

    unit_index = np.where(unit_ids == unit_id)[0][0]

    template = average_templates[unit_index]

    channel_index = np.argmax(
        np.max(np.abs(template), axis=0)
    )

    waveform = template[:, channel_index]

    template_peak = np.max(np.abs(waveform))
    si_amp = abs(metrics.loc[unit_id, "amplitude_median"])

    print(
        f"Unit {unit_id} | "
        f"SI amplitude_median={si_amp:.2f} uV | "
        f"SI template peak={template_peak:.2f} uV | "
        f"SI channel index={channel_index}"
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