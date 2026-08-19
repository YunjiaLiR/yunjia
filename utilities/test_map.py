import numpy as np
import os

folder_path = '20260804'
channel_map= np.load(os.path.join(folder_path, 'channel_map.npy')).flatten()

print(channel_map)
print("channel_map[0]=",channel_map[0])

import numpy as np
import pandas as pd
import os

folder_path = "20260804"

channel_map = np.load(
    os.path.join(folder_path, "channel_map.npy")
).flatten()

cluster_info = pd.read_csv(
    os.path.join(folder_path, "cluster_info.tsv"),
    sep="\t"
)

unit_id = 79

row = cluster_info.loc[cluster_info["cluster_id"] == unit_id]

phy_ch = int(row.iloc[0]["ch"])

print("Unit:", unit_id)
print("Phy/Kilosort ch:", phy_ch)
print("Mapped raw channel:", channel_map[phy_ch])