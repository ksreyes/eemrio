import os

os.chdir("/Users/divyasangaraju/Documents/GitHub/eemrio")
import pandas as pd
import codes.utils as utils
import codes.mrio as mrio
from codes.mrio import EE
from codes.utils import get_years, aggregate_sectors, convert_dtypes, ind_pattern

input = 'ee.parquet'
output = 'summary.parquet'

exp_dir= "/Users/divyasangaraju/Documents/Work/ADB/IO Publication/ee-output/"
years = get_years((exp_dir+input))

df = pd.DataFrame()

for year in years:

    ee = EE(exp_dir+input, year, by=['gas', 'sector'])
    G, N, K = ee.G, ee.N, len(ee.rows)
    emissions = ee.E.t().asvector()

    df_t = pd.DataFrame({
        't': year,
        's': ind_pattern(ee.country_inds(), repeat=N, tile=K),
        'i': ind_pattern(ee.sector_inds(), tile=G*K),
        'gas': ind_pattern(ee.rows[:][:, 0], repeat=G*N),
        'sector': ind_pattern(ee.rows[:][:, 1], repeat=G*N),
        'emissions': ee.E.asvector().data 
    })
    df = pd.concat([df, df_t], ignore_index=True)

summary = aggregate_sectors(
    table = 'df', 
    cols_index = ['t', 's', 'gas', 'sector'], 
    cols_to_sum = ['emissions']
)
summary = convert_dtypes(summary)
summary.to_parquet(exp_dir+output, index=False)
