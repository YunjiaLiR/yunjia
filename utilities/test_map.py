import numpy as np
import os

folder_path = '20260804'
channel_map= np.load(os.path.join(folder_path, 'channel_map.npy')).flatten()

print(channel_map)
print("channel_map[0]=",channel_map[0])