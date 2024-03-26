import numpy as np
import pandas as pd
import os
import re
import time

G, N, f = 73, 35, 5

output_mrio = f'mrio.parquet'
output_ee = f'ee.parquet'

files = [file for file in os.listdir(f'data/raw') if re.match(r'^ADB.*MRIO', file)]
files.sort()

mrio = pd.DataFrame()
ee = pd.DataFrame()

start = time.time()

for file in files:

    year = re.search('[0-9]{4}', file).group()
    table_t = pd.read_excel(f'data/raw/{file}', skiprows=5, header=[0,1])

    # Collapse MultiIndex headers into one
    table_t.columns = [f'{level_1}_{level_2}' for level_1, level_2 in table_t.columns]

    # Split
    mrio_t = table_t.iloc[ :G*N+8, 2:(5 + G*N + G*f)]
    ee_t = table_t.iloc[G*N+11: , 1:(4 + G*N + G*f)]

    # MRIO
    mrio_t = mrio_t[mrio_t.iloc[:, 1] != 'r60']
    mrio_t.columns.values[-1] = 'ToT'
    rowlabels = [
        f"{c}_{d}" if not (pd.isna(c) or c == 'ToT') else d 
        for c, d in zip(mrio_t.iloc[:, 0], mrio_t.iloc[:, 1])
    ]
    mrio_t.insert(2, 'si', rowlabels)
    mrio_t = mrio_t.iloc[:, 2:]
    mrio_t = mrio_t.replace(' ', 0)
    mrio_t.insert(0, 't', year)
    mrio = pd.concat([mrio, mrio_t], ignore_index=True)

    # EE
    ee_t = ee_t[-(ee_t.iloc[:, 2].isin(['Total']) | ee_t.iloc[:, 2].isna())]
    ee_t.insert(0, 't', year)
    ee_t.columns.values[1:4] = ['activity', 'gas', 'sector']
    ee = pd.concat([ee, ee_t], ignore_index=True)

    # Time check
    checkpoint = time.time()
    elapsed = checkpoint - start
    time_elapsed = f'{int(elapsed // 60)} mins {round(elapsed % 60, 1)} secs'

    print(f'\n{year} done. \nTime elapsed: {time_elapsed}.')

mrio['t'] = mrio['t'].astype(np.uint16)
ee['t'] = ee['t'].astype(np.uint16)
mrio.to_parquet(f'data/{output_mrio}', index=False)
ee.to_parquet(f'data/{output_ee}', index=False)