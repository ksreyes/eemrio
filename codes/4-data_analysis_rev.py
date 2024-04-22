#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 23:31:04 2024

@author: divyasangaraju
"""

import pandas as pd 
import os
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import re




########################################################################################

## IMPORT EE-EMISSIONS SUMMARISED DATA IN (KEN'S DATA) 

with open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/1-preprocess-eemrios.py', 'r') as file:
    code = file.read()
exec(code)

with open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/2-summary.py', 'r') as file:
    code = file.read()
exec(code)

with open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/3-eby.py', 'r') as file:
    code = file.read()
exec(code)

########################################################################################

## IMPORT MRIO TABLES RAW IN 


years = [2017,2018,2019,2020,2021]
mriot_years ={}

for i in years:
    raw = pd.read_excel("/Users/divyasangaraju/Library/CloudStorage/OneDrive-SharedLibraries-AsianDevelopmentBank/SDIU - GVC/Climate Change/IO Publication 2024/EE-MRIOTs_as of March 2024/ADB-EE-MRIO-"+str(i)+".xlsx")
    if i==2017 or i==2018 or i==2019:
        raw=raw[2:]
        raw = raw.drop(columns=raw.columns[[0, 1]])
        first_instance_index = raw.columns.get_loc(raw.columns[raw.iloc[1].eq("TOTAL")][0])+1
        raw = raw.iloc[:, 0:first_instance_index]
        raw = raw.rename(columns={raw.columns[0]: 's'})
        raw = raw.rename(columns={raw.columns[1]: 's_sector'})
        raw = raw.sort_values(by='s')
        raw['Unique_ID'] = pd.factorize(raw['s'])[0]
        raw = raw.sort_values(by='Unique_ID')
        raw = raw.set_index('Unique_ID').reset_index(drop=False)
        mriot_years[i] = raw
        print(str(i)+" Complete")

    else:
        raw=raw[1:]
        raw = raw.drop(columns=raw.columns[[0, 1]])
        first_instance_index = raw.columns.get_loc(raw.columns[raw.iloc[2].eq("TOTAL")][0])+1
        raw = raw.iloc[:, 0:first_instance_index]
        raw = raw.rename(columns={raw.columns[0]: 's'})
        raw = raw.rename(columns={raw.columns[1]: 's_sector'})
        raw = raw.sort_values(by='s')
        raw['Unique_ID'] = pd.factorize(raw['s'])[0]
        raw = raw.sort_values(by='Unique_ID')
        raw = raw.set_index('Unique_ID').reset_index(drop=False)
        mriot_years[i] = raw
        print(str(i)+" Complete")

        




########################################################################################

## Compiling EE tables 

ee_table = {}

for i in years:
    raw = pd.read_excel("/Users/divyasangaraju/Library/CloudStorage/OneDrive-SharedLibraries-AsianDevelopmentBank/SDIU - GVC/Climate Change/IO Publication 2024/EE-MRIOTs_as of March 2024/ADB-EE-MRIO-"+str(i)+".xlsx")
    gas_rows = raw.index[raw.iloc[:, 2] == 'Gas'].tolist()[0]
    temp=raw[gas_rows:]
    new_row_dict = raw.iloc[[4, 5]]
    ee_t = pd.concat([new_row_dict, temp], ignore_index=True)
    ee_table[i] = ee_t
    print(str(i)+" Complete")


## Compute Sum of EE Emission over years 
merged_ee = {}
for i in years:
    test1 = ee_table[i]
    test2 = test1.T
    test2 = test2.rename(columns={test2.columns[0]:'mrio_code'})
    test2 = test2.groupby('mrio_code').sum()
    test2.columns = test1.iloc[1:,1]
    test2 = test2.T.reset_index()
    test2 = test2.rename(columns={test2.columns[0]:'variable'})
    test2 = test2.groupby('variable').sum()
    test2 = test2.reset_index()
    test2 = test2.melt(id_vars='variable', var_name='mrio_code', value_name='Value')
    merged_ee[i] = test2 
    test2=""


## Combine data into 1 dataset

    print(str(i) +" Complete")
########################################################################################

## Compute GDP from each year
id_list = range(0,max(raw['Unique_ID']))
GDP_years = pd.DataFrame()
GDP_years['s']= raw['s'].unique()
sum_values = pd.DataFrame()
for i in years:
    test = mriot_years[i]
    sum_values = pd.DataFrame(test[(test['s_sector'].isin(['r99','r64']))].sum())
    sum_values.index = test.iloc[4]
    sum_values=sum_values.reset_index()
    sum_values = sum_values.rename(columns={sum_values.columns[0]: 's'})
    sum_values = sum_values.groupby('s')[0].sum()
    temp = pd.merge(GDP_years, sum_values, on='s', how='outer')
    GDP_years[i] = temp[0]
    print(str(i) +" Complete")
    temp = ""
    # else:        
    #     sum_values.index = test.iloc[9, :]
    #     sum_values=sum_values.reset_index()
    #     sum_values = sum_values.rename(columns={sum_values.columns[0]: 's'})
    #     sum_values = sum_values.groupby('s')[0].sum()
    #     temp = pd.merge(GDP_years, sum_values, on='s', how='outer')
    #     GDP_years[i] = temp[0]
    #     print(str(i) +" Complete")
    #     temp = ""

## Compute GDP Growth Rates Across Years

GDP_years = GDP_years.set_index('s')
GDP_years_pctchg = GDP_years.pct_change(axis=1) * 100
GDP_years_pctchg['2017_2021'] = (GDP_years[2021]/GDP_years[2017])-1

########################################################################################

## Compute Total Energy Emissions from each year
energy_t=pd.DataFrame()
energy_t = eby[eby['sector']=='Energy']
preserved_columns = energy_t[['r','i','s']]
energy_t = energy_t.pivot(columns='t',values='emissions')
energy_t = pd.concat([preserved_columns, energy_t], axis=1)
energy_t = energy_t.groupby('s').sum()
energy_t =energy_t.drop(energy_t.columns[[0, 1]], axis=1)
energy_t_pctchg = energy_t.pct_change(axis=1) * 100
energy_t_pctchg['2017_2021'] = (energy_t[2021]/energy_t[2017])-1

energy_t_pctchg = energy_t_pctchg.reset_index()
energy_t_pctchg = energy_t_pctchg.rename(columns={'s': 'mrio'})
country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')
energy_t_pctchg = pd.merge(energy_t_pctchg, country_dict, on='mrio', how='inner')
energy_t_pctchg = energy_t_pctchg.rename(columns={'mrio_code': 's'})


########################################################################################
########################################################################################
## Plot Scatterplot of GDP Change with that of Change in Energy Emissions 
 
### Merge databases together
plt_dset=pd.DataFrame()
plt_dset['s'] = energy_t_pctchg['s']
plt_dset = pd.merge(GDP_years_pctchg, plt_dset, on='s', how='outer')
plt_dset = pd.merge(energy_t_pctchg, plt_dset, on='s', how='outer')


### Match with Region Types 

region_dict = pd.read_excel("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/ADB members - MRIO - ISO 1.xlsx")
region_dict = region_dict.rename(columns={'MRIO': 's'})
plt_dset = pd.merge(plt_dset,region_dict,on='s',how='outer')
plt_dset['Region'] = pd.get_dummies(plt_dset['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
plt_dset['Region'] = plt_dset['Region'].replace(replacement_map)


plt_dset_filt = plt_dset
plt_dset_filt = plt_dset_filt[plt_dset_filt['2017_2021_x']>-20]
plt_dset_filt['2017_2021_x'] = pd.to_numeric(plt_dset_filt['2017_2021_x'], errors='coerce')
plt_dset_filt['2017_2021_y'] = pd.to_numeric(plt_dset_filt['2017_2021_y'], errors='coerce')

fig, ax = plt.subplots() 
colors = ['red', 'royalblue']
markers = ['.', '+']

for i, value in enumerate(plt_dset_filt.Region.unique()):
    ax = sns.regplot(x="2017_2021_x", y="2017_2021_y", ax=ax,
                     color=colors[i],
                     marker=markers[i], 
                     data=plt_dset_filt[plt_dset_filt.Region == value],
                     label=value, fit_reg=True)
plt.xlabel('GDP % Change between 2017 and 2021')
plt.ylabel('Energy Emissions % Change between 2017 and 2021')
ax.legend(loc='best') 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
# Save the plot to a file
plt.savefig("scatter_em_gdp.png")
plt.close()



########################################################################################
########################################################################################

# Largest Energy Producers in the Region 

energy_supp = pd.read_excel("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/IEA_data_summ.xlsx")

########################################################################################
########################################################################################
## Plot Total Energy Producer by Country 
year = 2021
energy_supp = energy_supp.sort_values(by=str(year),ascending=False)
energy_supp = energy_supp.rename(columns={energy_supp.columns[0]: 'mrio_name'})
energy_supp['mrio_name']=energy_supp['mrio_name'].str.title()
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Australi', 'Australia',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('China', '''People's Republic of China''',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Switland', 'Switzerland',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Vietnam', 'Viet Nam',regex=True)                                            
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Usa', 'United States',regex=True)                                                                                                      
energy_supp = energy_supp.replace({'mrio_name': {'Uk': 'United Kingdom'}})                                              
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Usa', 'United States',regex=True)    
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Turkey', 'Trkiye',regex=True)                                                                                                      
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Taipei', 'Taipei, China',regex=True)     
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Taipei,China', 'Taipei, China',regex=True)                                                                                                                            
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Ssudan', 'Sudan',regex=True)                                                                                                      
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Srilanka', 'Sri Lanka',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Slovakia', 'Slovak Republic',regex=True)                                                                                                      
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Saudiarabi', 'Saudi Arabia',regex=True)                                                                                                
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Russia', 'Russian Federation',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Phillippine', 'Philippines',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Nz', 'New Zealand',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Nethland', 'Netherlands',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Luxembou', 'Luxembourg',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Lao', '''Lao People's Democratic Republic''',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Kyrgyz Republic', 'Kyrgyzstan',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Korea', 'Republic of Korea',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Czech', 'Czech Republic',regex=True)
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Hongkong', 'Hong Kong, China',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Brunei', 'Brunei Darussalam',regex=True) 
energy_supp['mrio_name']=energy_supp['mrio_name'].replace('Uae', 'United Arab Emirates',regex=True) 
energy_supp = pd.merge(energy_supp, country_dict, on='mrio_name', how='outer')
energy_supp = energy_supp.dropna(subset=['mrio_code'])
energy_supp_filt = energy_supp[0:20]


plt.figure(figsize=(40,20))  # Adjust the size as needed
ax=sns.barplot(x=energy_supp_filt.mrio_code,y=energy_supp_filt[str(year)], data=energy_supp_filt)
plt.xticks(rotation=90)
plt.xlabel('Economy', fontsize=30)
plt.ylabel('Energy Producer')
plt.title(year,fontsize=40)
for index, value in enumerate(energy_supp_filt[str(year)]):
    formatted_value = '{:,.0f}'.format(value)  # Format value with comma as thousand separators
    plt.text(index, value - 900000, formatted_value, ha='center', va='bottom', fontsize=20, color='black', weight='bold',rotation=90)
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
plt.savefig("energy_producer"+str(year)+".png")
plt.show()
# Close the plot
plt.close()


########################################################################################
########################################################################################

# Largest Energy Emissions in the Region 
country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')
energy_t = energy_t.reset_index()
energy_t = energy_t.rename(columns={'s': 'mrio'})
energy_t= pd.merge(energy_t, country_dict, on='mrio', how='inner')

########################################################################################
########################################################################################
## Plot Total Energy Emissions by Country 
year = 2021
energy_t = energy_t.sort_values(by=year,ascending=False)
energy_t = energy_t[energy_t.mrio_code!="ROW"]
energy_t_filt = energy_t[0:20]
plt.figure(figsize=(40,20))  # Adjust the size as needed
ax=sns.barplot(x=energy_t_filt.mrio_code,y=energy_t_filt[year], data=energy_t_filt)
plt.xticks(rotation=90)
plt.xlabel('Economy', fontsize=30)
plt.ylabel('Energy Emissions Produced')
plt.title(year,fontsize=40)
for index, value in enumerate(energy_t_filt[year]):
    formatted_value = '{:,.0f}'.format(value)  # Format value with comma as thousand separators
    plt.text(index, value - 900000, formatted_value, ha='center', va='bottom', fontsize=20, color='black', weight='bold',rotation=90)
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
plt.savefig("energy_emissions"+str(year)+".png")
plt.show()
# Close the plot
plt.close()


########################################################################################
########################################################################################

## Plot Total Energy Emissions Against Total Energy Supply by Country 
year = 2020
energy_t_supp= pd.merge(energy_t, energy_supp, on='mrio_code', how='inner')
energy_t_supp_yr =energy_t_supp[['mrio_code', year, str(year)]]
region_dict = region_dict.rename(columns={'s': 'mrio_code'})
energy_t_supp_yr = pd.merge(energy_t_supp_yr,region_dict,on='mrio_code',how='outer')
energy_t_supp_yr['Region'] = pd.get_dummies(energy_t_supp_yr['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
energy_t_supp_yr['Region'] = energy_t_supp_yr['Region'].replace(replacement_map)

fig, ax = plt.subplots() 
colors = ['red', 'royalblue']
markers = ['.', '+']
energy_t_supp_yr = energy_t_supp_yr.rename(columns={str(year): 'energy_t'})
energy_t_supp_yr = energy_t_supp_yr.rename(columns={year: 'energy_s'})


for i, value in enumerate(energy_t_supp_yr.Region.unique()):
    ax = sns.regplot(x='energy_s', y='energy_t', ax=ax,
                     color=colors[i],
                     marker=markers[i], 
                     data=energy_t_supp_yr[energy_t_supp_yr.Region == value],
                     label=value, fit_reg=True,robust=True)

plt.ylim(0, 7e8)  # Adjust the range as needed
plt.xlim(0, 5e7)  # Adjust the range as needed
plt.xlabel('Energy Supply')
plt.ylabel('Energy Emissions')
plt.title(year)
ax.legend(loc='best') 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
# Save the plot to a file
plt.savefig("scatter_em_supp_"+str(year)+".png")
plt.close()

########################################################################################


# Output Consolidation across Years
output_years = pd.DataFrame()
output_years['s'] = raw['s'].unique()
for i in years:
    temp = mriot_years[i]
    row_index = 3  # Row index to search for the text
    search_text = 'TOTAL'  # Text to search for in the row
    column_to_drop = temp.columns[temp.iloc[row_index] == search_text][0]
    temp.drop(columns=column_to_drop)
    start_column_index = temp.columns.get_loc('s_sector') + 1
    sum_values = temp.groupby('s').apply(lambda x: x.iloc[:, start_column_index:].sum(axis=1))
    sum_values = sum_values.reset_index()
    sum_values = sum_values.rename(columns={sum_values.columns[2]: str(i)})
    sum_values.drop(columns='level_1', inplace=True)
    sum_values = sum_values.groupby('s').sum()
    sum_values = sum_values.reset_index()
    output_years=pd.merge(output_years, sum_values, on='s', how='outer')
    print(str(i)+" Complete")

  

########################################################################################
########################################################################################

#Plot Output against Emissions 
year = 2021
    
output_years = output_years.rename(columns={'s': 'mrio_code'})
output_years_em = pd.merge(output_years,energy_t, on='mrio_code', how='outer') 
output_years_em = pd.merge(output_years_em,region_dict,on='mrio_code',how='outer')
output_years_em['Region'] = pd.get_dummies(output_years_em['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
output_years_em['Region'] = output_years_em['Region'].replace(replacement_map)
output_years_em = output_years_em.rename(columns={str(year): 'output'})
output_years_em = output_years_em.rename(columns={year: 'energy_t'})

fig, ax = plt.subplots() 
colors = ['red', 'royalblue']
markers = ['.', '+']


for i, value in enumerate(output_years_em.Region.unique()):
    ax = sns.regplot(x='output', y='energy_t', ax=ax,
                     color=colors[i],
                     marker=markers[i], 
                     data=output_years_em[output_years_em.Region == value],
                     label=value, fit_reg=True,robust=True)

plt.ylim(0, 6e7)  # Adjust the range as needed
plt.xlim(0, 11e7)  # Adjust the range as needed
plt.xlabel('Total Output')
plt.ylabel('Energy Emissions')
plt.title(year)
ax.legend(loc='best') 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
# Save the plot to a file
plt.savefig("scatter_output_em_"+str(year)+".png")
plt.close()


########################################################################################
# Calculate Output Change 
output_years_pctchg = pd.DataFrame()
output_years_pctchg['s']=output_years['mrio_code']
output_years_pctchg = output_years_pctchg.set_index('s')
output_years_filt = output_years.set_index('mrio_code')
output_years_pctchg = output_years_filt.pct_change(axis=1) * 100

output_years_pctchg['2017_2021'] = (output_years_filt['2021']/output_years_filt['2017'])-1

# Calculate Emissions Change 
energy_t_pctchg = pd.DataFrame()
energy_t_pctchg['s']=energy_t['mrio_code']
energy_t_pctchg = energy_t_pctchg.set_index('s')
energy_t_filt = energy_t.set_index('mrio_code')
energy_t_filt = energy_t_filt[[2017,2018,2019,2020,2021]]
energy_t_pctchg = energy_t_filt.pct_change(axis=1) * 100
energy_t_pctchg['2017_2021'] = (energy_t_filt[2021]/energy_t_filt[2017])-1

# Merge Output Change and Emissions Change Data

output_em_chg = pd.merge(output_years_pctchg,energy_t_pctchg,on='mrio_code')

output_em_chg = output_em_chg.reset_index()

########################################################################################
########################################################################################
#Plot Output Change against Emissions Change


region_dict = pd.read_excel("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/ADB members - MRIO - ISO 1.xlsx")
region_dict = region_dict.rename(columns={'MRIO': 'mrio_code'})
output_em_chg = pd.merge(output_em_chg,region_dict,on='mrio_code',how='outer')
output_em_chg['Region'] = pd.get_dummies(output_em_chg['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
output_em_chg['Region'] = output_em_chg['Region'].replace(replacement_map)


output_em_chg_filt = output_em_chg
output_em_chg_filt['2017_2021_x'] = pd.to_numeric(output_em_chg_filt['2017_2021_x'], errors='coerce')
output_em_chg_filt['2017_2021_y'] = pd.to_numeric(output_em_chg_filt['2017_2021_y'], errors='coerce')

fig, ax = plt.subplots() 
colors = ['royalblue', 'red']
markers = ['.', '+']

for i, value in enumerate(output_em_chg_filt.Region.unique()):
    ax = sns.regplot(x="2017_2021_x", y="2017_2021_y", ax=ax,
                     color=colors[i],
                     marker=markers[i], 
                     data=output_em_chg_filt[output_em_chg_filt.Region == value],
                     label=value, fit_reg=True)
plt.xlabel('Output % Change between 2017 and 2021')
plt.ylabel('Energy Emissions % Change between 2017 and 2021')
ax.legend(loc='best') 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
# Save the plot to a file
plt.savefig("scatter_em_output_chg.png")
plt.close()

########################################################################################
# Import Energy Generation Data Split 
energy_supp_split = pd.read_excel("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/IEA_data_summ2.xlsx")
energy_supp_split = energy_supp_split[energy_supp_split['V2']!="TOTAL"]
energy_supp_split_sum = energy_supp_split.melt(id_vars=['RE','V1'], var_name='Year', value_name='Value')
pivot_df = energy_supp_split_sum.pivot_table(index=['Year','V1'],columns='RE', values='Value', aggfunc='sum')

# Calculate Proportion of Renewable Energy 
pivot_df[1] = pd.to_numeric(pivot_df[1], errors='coerce')
pivot_df[0] = pd.to_numeric(pivot_df[0], errors='coerce')

pivot_df['renew_prop'] = pivot_df[1]/(pivot_df[1]+pivot_df[0])*100

pivot_df  = pivot_df[['renew_prop']]

pivot_df  = pivot_df.reset_index()
pivot_df = pivot_df.pivot(index='V1', columns='Year', values='renew_prop')
pivot_df_pctchg = pivot_df.apply(pd.to_numeric, errors='coerce')
pivot_df_pctchg = pivot_df.pct_change(axis=1) * 100
pivot_df_pctchg['2017_2021'] = (pivot_df_pctchg[str(2021)]/pivot_df_pctchg[str(2017)])-1
pivot_df_pctchg = pivot_df_pctchg.reset_index()
pivot_df_pctchg = pivot_df_pctchg.rename(columns={'V1': 'mrio_name'})


year = 2018
pivot_df_pctchg = pivot_df_pctchg.sort_values(by=str(year),ascending=False)
pivot_df_pctchg = pivot_df_pctchg.rename(columns={'V1': 'mrio_name'})
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].str.title()
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Australi', 'Australia',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('China', '''People's Republic of China''',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Switland', 'Switzerland',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Vietnam', 'Viet Nam',regex=True)                                            
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Usa', 'United States',regex=True)                                                                                                      
pivot_df_pctchg = pivot_df_pctchg.replace({'mrio_name': {'Uk': 'United Kingdom'}})                                              
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Usa', 'United States',regex=True)    
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Turkey', 'Trkiye',regex=True)                                                                                                      
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Taipei', 'Taipei, China',regex=True)     
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Taipei,China', 'Taipei, China',regex=True)                                                                                                                            
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Ssudan', 'Sudan',regex=True)                                                                                                      
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Srilanka', 'Sri Lanka',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Slovakia', 'Slovak Republic',regex=True)                                                                                                      
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Saudiarabi', 'Saudi Arabia',regex=True)                                                                                                
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Russia', 'Russian Federation',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Phillippine', 'Philippines',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Nz', 'New Zealand',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Nethland', 'Netherlands',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Luxembou', 'Luxembourg',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Lao', '''Lao People's Democratic Republic''',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Kyrgyz Republic', 'Kyrgyzstan',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Korea', 'Republic of Korea',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Czech', 'Czech Republic',regex=True)
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Hongkong', 'Hong Kong, China',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Brunei', 'Brunei Darussalam',regex=True) 
pivot_df_pctchg['mrio_name']=pivot_df_pctchg['mrio_name'].replace('Uae', 'United Arab Emirates',regex=True) 
pivot_df_pctchg = pd.merge(pivot_df_pctchg, country_dict, on='mrio_name', how='outer')
pivot_df_pctchg = pivot_df_pctchg.dropna(subset=['mrio_code'])
region_dict = pd.read_excel("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/ADB members - MRIO - ISO 1.xlsx")
region_dict = region_dict.rename(columns={'MRIO': 'mrio_code'})
pivot_df_pctchg = pd.merge(pivot_df_pctchg,region_dict,on='mrio_code',how='outer')
pivot_df_pctchg['Region'] = pd.get_dummies(pivot_df_pctchg['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
pivot_df_pctchg['Region'] = pivot_df_pctchg['Region'].replace(replacement_map)

#Merge emissions into the data 

renew_em= pd.merge(pivot_df_pctchg, energy_t_pctchg, on='mrio_code', how='inner')
renew_em = renew_em.rename(columns={str(year): 'renew_prop'})
renew_em = renew_em.rename(columns={year: 'energy_t_chg'})
renew_em = renew_em[renew_em['2017_2021_x']>-20]


########################################################################################
########################################################################################
#Plot Renew Prop against Energy Emissions
fig, ax = plt.subplots() 
colors = ['red', 'royalblue']
markers = ['.', '+']


for i, value in enumerate(renew_em.Region.unique()):
    ax = sns.regplot(x='2017_2021_x', y='2017_2021_y', ax=ax,
                     color=colors[i],
                     marker=markers[i], 
                     data=renew_em[renew_em.Region == value],
                     label=value, fit_reg=True,robust=True)
plt.xlabel('Renewable Proportion (%)')
plt.ylabel('Energy Emissions Chg')
plt.title("2017-2021 Chg")
ax.legend(loc='best') 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
# Save the plot to a file
plt.savefig("scatter_renewprop_emchg"+".png")
plt.close()


########################################################################################
# Global Powerplant Database

power_plant = pd.read_csv("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/global_power_plant_database_v_1_3/global_power_plant_database.csv")
power_plant.dropna(subset=['commissioning_year'], inplace=True)
power_plant['commissioning_year'] = power_plant['commissioning_year'].round()
power_plant_count = power_plant.groupby(['country','commissioning_year','primary_fuel']).count()
power_plant_count = power_plant_count[['name']]
power_plant_count=power_plant_count.reset_index()
power_plant_count['RE']  = 0
power_plant_count.loc[(power_plant_count['primary_fuel'] =='Hydro') | (power_plant_count['primary_fuel'] =='Solar')|(power_plant_count['primary_fuel'] =='Nuclear')|(power_plant_count['primary_fuel'] =='Wind')|(power_plant_count['primary_fuel'] =='Biomass')|(power_plant_count['primary_fuel'] =='Waste')|(power_plant_count['primary_fuel'] =='Geothermal'), 'RE'] = 1


## Split to renewable & Non Renwable dataframes

power_plant_ren = power_plant_count[power_plant_count['RE']==1]
power_plant_ren = power_plant_ren.groupby(['country','commissioning_year','RE']).sum()
power_plant_ren=power_plant_ren.reset_index()
power_plant_ren = power_plant_ren.pivot(index='country',columns='commissioning_year',values='name')
power_plant_ren = power_plant_ren.fillna(0)
power_plant_ren= power_plant_ren.cumsum(axis=1)
power_plant_ren=power_plant_ren.reset_index()
power_plant_ren = power_plant_ren.rename(columns={'country': 'mrio_code'})
power_plant_ren = pd.merge(power_plant_ren,region_dict,on='mrio_code',how='outer')
power_plant_ren['Region'] = pd.get_dummies(power_plant_ren['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
power_plant_ren['Region'] = power_plant_ren['Region'].replace(replacement_map)
power_plant_ren_reg = power_plant_ren.groupby(['Region']).sum()

power_plant_nonren = power_plant_count[power_plant_count['RE']==0]
power_plant_nonren = power_plant_nonren.groupby(['country','commissioning_year','RE']).sum()
power_plant_nonren=power_plant_nonren.reset_index()
power_plant_nonren = power_plant_nonren.pivot(index='country',columns='commissioning_year',values='name')
power_plant_nonren = power_plant_nonren.fillna(0)
power_plant_nonren= power_plant_nonren.cumsum(axis=1)
power_plant_nonren=power_plant_nonren.reset_index()
power_plant_nonren = power_plant_nonren.rename(columns={'country': 'mrio_code'})
power_plant_nonren = pd.merge(power_plant_nonren,region_dict,on='mrio_code',how='outer')
power_plant_nonren['Region'] = pd.get_dummies(power_plant_nonren['ADB'])
replacement_map = {1: 'ADB Member', 0: 'Non-ADB Member'}
power_plant_nonren['Region'] = power_plant_nonren['Region'].replace(replacement_map)

power_plant_nonren_reg = power_plant_nonren.groupby(['Region']).sum()
temp1 = power_plant_nonren_reg.T.reset_index()
temp2= power_plant_ren_reg.T.reset_index()

## Merge two dataseries 
power_plant_merge_reg=pd.merge(temp1,temp2,on='index',how='inner')
power_plant_merge_reg = power_plant_merge_reg.set_index('index')

power_plant_merge_reg.columns = power_plant_merge_reg.columns.str.split('_', expand=True)

# Unstack only level 0
power_plant_merge_plt = power_plant_merge_reg.stack().reset_index()
power_plant_merge_plt = power_plant_merge_plt[power_plant_merge_plt['index']>1995]
power_plant_merge_plt['index'] = power_plant_merge_plt['index'].astype(int)

########################################################################################
########################################################################################

#Plot No of energy plants over time 
power_plant_merge_plt_lng = power_plant_merge_plt.melt(id_vars=['index','level_1'], var_name='Variable', value_name='Value')
power_plant_merge_plt_lng =power_plant_merge_plt_lng[power_plant_merge_plt_lng['Variable']=='ADB Member']
power_plant_merge_plt_lng.loc[power_plant_merge_plt_lng['level_1'] == "x", 'level_1'] = "Non-Renewable Energy Plant"
power_plant_merge_plt_lng.loc[power_plant_merge_plt_lng['level_1'] == "y", 'level_1'] = "Renewable Energy Plant"
plt.figure(figsize=(10, 6))
ax = sns.barplot(data=power_plant_merge_plt_lng, x='index', y='Value', hue='level_1',ci=None)


# Add labels and title
plt.xlabel('Year')
plt.ylabel('Nummber of Energy Generation Plants')
plt.title('ADB Members Cummulation No of Energy Generation Plants by Plant Commencment Year')
plt.xticks(rotation=90, ha='right')
ax.legend(loc='best') 
plt.tight_layout() 
os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")
plt.show()
plt.savefig("barplot_energyplant_ADB"+".png")
plt.close()

########################################################################################

## Energy Consumption Emissions 













########################################################################################

# Bilateral Linkage Chart
import pandas as pd
import holoviews as hv
from holoviews import opts, dim

year = 2021

hv.extension('matplotlib')
hv.output(fig='svg', size=250)

eby_energy= eby[eby['sector']=='Energy']

eby_energy['emissions_inmil'] = eby_energy['emissions']/1000000
e_producer_consumer = pd.DataFrame(eby_energy.groupby(['r','s','t'])['emissions_inmil'].sum()).sort_values(by='emissions_inmil',ascending=False).reset_index()
e_producer_consumer=e_producer_consumer.dropna()
e_producer_consumer = e_producer_consumer[e_producer_consumer.t ==year]
e_producer_consumer=e_producer_consumer.rename(columns={'s':'source','r':'target','emissions_inmil':'value','t':'node'})
e_producer_consumer = e_producer_consumer[['target','source','value']]

country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')
country_dict.set_index('mrio')
node = country_dict[['mrio_code','mrio']]
node=node.rename(columns={'mrio_code':'name','mrio':'index'})


########################################################################################
########################################################################################
# Bilateral Linkage Chart 
nodes = hv.Dataset(node.reset_index(), 'index')
e_producer_consumer['value'].fillna(0, inplace=True)
#customization of chart
img = hv.Chord((e_producer_consumer, nodes)).select(value=(0.3, None)).opts(
    opts.Chord(fontsize={'title': 25, 'labels': 15, 'xticks': 20, 'yticks': 20},title="Bilateral Emission Flows " + str(year),cmap='Category10', node_size=10,sublabel_size=35, cbar_width=5,edge_cmap='Category10', edge_color=dim('source').astype(str), labels='name', node_color=dim('index').astype(str)))
hv.render(img)


