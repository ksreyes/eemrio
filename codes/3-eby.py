import pandas as pd
import codes.utils as utils
import codes.mrio as mrio
from codes.mrio import MRIO, EE
from codes.utils import get_years, ind_pattern, aggregate_sectors, convert_dtypes



input_mrio = 'mrio.parquet'
input_ee = 'ee.parquet'
output = 'eby.parquet'
exp_dir= "/Users/divyasangaraju/Documents/Work/ADB/IO Publication/ee-output/"
years = get_years((exp_dir+input_ee))

df = pd.DataFrame()

for year in years:
        
    mrio = MRIO(exp_dir+input_mrio, year, full=True)
    ee = EE(exp_dir+input_ee, year, by = 'sector')
    G, N = mrio.G, mrio.N

    sector = ee.rows
    K = len(ee.rows)
    e = ee.E @ (1/mrio.x).diag() #emission coefficient
    BY = mrio.B @ mrio.Y
    EBY = e.diagstack() @ BY

    df_t = pd.DataFrame({
        't': year,
        'sector': ind_pattern(sector, repeat=G*N, tile=G),
        's': ind_pattern(mrio.country_inds(), repeat=N, tile=K*G),
        'i': ind_pattern(mrio.sector_inds(), tile=G*K*G),
        'r': ind_pattern(mrio.country_inds(), repeat=G*N*K),
        'emissions': EBY.asvector().data,
    })
    df = pd.concat([df, df_t], ignore_index=True)

    print(f'{year} done.')

eby = aggregate_sectors(
    table = 'df',
    cols_index = ['t', 'sector', 's', 'r'],
    cols_to_sum = ['emissions']
)
eby = convert_dtypes(eby)
eby.to_parquet(exp_dir+output, index=False)