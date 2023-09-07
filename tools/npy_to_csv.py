import os
import numpy as np
import pandas as pd
from tqdm import tqdm

directory_path = "val_set/annotations/"
file_list = sorted([filename for filename in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, filename))])
data  = []

for i in tqdm(range(0, len(file_list), 4), desc="Processing Files"):
    group = file_list[i:i + 4]
    filepath = f"val_set/images/{group[1].strip('_exp.npy')}.jpg"

    arl = np.load(f"val_set/annotations/{group[0]}")
    exp = np.load(f"val_set/annotations/{group[1]}")
    lnd = np.load(f"val_set/annotations/{group[2]}")
    val = np.load(f"val_set/annotations/{group[3]}")

    array = [filepath, exp, val, arl, lnd]
    data.append(array)

df = pd.DataFrame(data, columns=['Filepath','Expression', 'Valence', 'Arousal', 'Landmark'])
df.to_csv("AffectNet_VAL.csv", index=False) 
