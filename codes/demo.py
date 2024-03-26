import numpy as np
import pandas as pd

countries = ['C', 'J', 'U']
sectors = [1, 2]
si = [f'{s}{i}' for s in countries for i in sectors] + ['VA', 'ToT']

mrio = pd.DataFrame({
    't': 2020,
    'si': si,
    f'{si[0]}': [0] * 6 + [3, 3],
    f'{si[1]}': [0] * 6 + [1, 1],
    f'{si[2]}': [0] * 6 + [2, 2],
    f'{si[3]}': [2] + [0] * 5 + [1.5, 3.5],
    f'{si[4]}': [0] * 6 + [2, 2],
    f'{si[5]}': [0] * 6 + [1, 1],
    'Cf': [1, 1] + [0] * 4 + [np.nan] * 2,
    'Jf': [0] * 2 + [2, 1] + [0] * 2 + [np.nan] * 2,
    'Uf': [0] * 3 + [2.5, 2, 1] + [np.nan] * 2,
    'ToT': [3, 1, 2, 3.5, 2, 1] + [np.nan] * 2,
})

mrio.to_parquet('data/dummy_mrio.parquet', index=False)
