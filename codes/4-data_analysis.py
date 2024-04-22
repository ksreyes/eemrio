import pandas as pd import osimport subprocessimport seaborn as snsimport matplotlib.pyplot as pltwith open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/1-preprocess-eemrios.py', 'r') as file:    code = file.read()exec(code)with open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/2-summary.py', 'r') as file:    code = file.read()exec(code)with open('/Users/divyasangaraju/Documents/GitHub/eemrio/codes/3-eby.py', 'r') as file:    code = file.read()exec(code)######################################################################################### MRIO Raw Data year = 2019mriot_year = pd.read_excel("/Users/divyasangaraju/Library/CloudStorage/OneDrive-SharedLibraries-AsianDevelopmentBank/SDIU - GVC/Climate Change/IO Publication 2024/EE-MRIOTs_as of March 2024/ADB-EE-MRIO-"+str(year)+".xlsx")[4:]del mriot_year[mriot_year.columns[0]]del mriot_year[mriot_year.columns[0]]mriot_year = mriot_year.rename(columns={mriot_year.columns[0]: 's'})mriot_year = mriot_year.rename(columns={mriot_year.columns[1]: 's_sector'})# Largest Energy Supplier mriot_year_df = mriot_year[mriot_year['s_sector']=='c17']mriot_year_df_energyproduced = mriot_year_df.drop(mriot_year_df.columns[:2], axis=1)mriot_year_df_energysum = mriot_year_df_energyproduced.sum(axis=1)mriot_year_df_energysum_fin = pd.concat([mriot_year_df['s'], mriot_year_df_energysum], axis=1)plt.figure(figsize=(20,10))  # Adjust the size as neededmriot_year_df_energysum_fin = mriot_year_df_energysum_fin.sort_values(by=0,ascending=False)mriot_year_df_energysum_fin_plt =mriot_year_df_energysum_fin[0:20] ax=sns.barplot(x=mriot_year_df_energysum_fin_plt.s,y=mriot_year_df_energysum_fin_plt[0], data=mriot_year_df_energysum_fin_plt)plt.xticks(rotation=90)plt.xlabel('Economy')plt.ylabel('Energy Producer Value (in Million)')plt.title("Energy Producer ("+str(year)+")")for p in ax.patches:    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90, weight='bold')os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")# Save the plot to a fileplt.savefig("producer"+str(year)+".png")# Close the plotplt.close()### Emission Producer plt.clf()eby_energy = eby[eby['sector']=='Energy']eby_energy['emissions'] = pd.to_numeric(eby_energy['emissions'])eby_energy['emissions_inmil'] = eby_energy['emissions']/1000000e_producer = pd.DataFrame(eby_energy.groupby(['s','t'])['emissions_inmil'].sum()).sort_values(by='emissions_inmil',ascending=False).reset_index()e_producer=e_producer.pivot(index='s', columns='t', values='emissions_inmil').reset_index()e_producer=e_producer.rename(columns={'s':'mrio'})country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')e_producer_cname = pd.merge(e_producer, country_dict, on='mrio', how='inner')plt.figure(figsize=(20,10))  # Adjust the size as needede_producer_cname = e_producer_cname.sort_values(by=year,ascending=False)e_producer_cname_plt = e_producer_cname[0:20]ax=sns.barplot(x=e_producer_cname_plt.mrio_name,y=e_producer_cname_plt[year], data=e_producer_cname_plt)plt.xticks(rotation=90)plt.xlabel('Economy')plt.ylabel('Emission Value (in Million)')plt.title("Energy Emission Producer"+str(year))plt.ylim(0, 80)  # Adjust the range as neededfor p in ax.patches:    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90, weight='bold')os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")plt.savefig("energyemission"+str(year)+"_prod.png")# Close the plotplt.close()### Emission Produced per unit of Energy Supplied e_producer_cname_filt = pd.DataFrame(e_producer_cname['mrio_code'])e_producer_cname_filt[year] = e_producer_cname[year]temp =mriot_year_df_energysum_fin.rename(columns={'s':'mrio_code'})inner_join_df = pd.merge(e_producer_cname_filt, temp, on='mrio_code', how='inner')inner_join_df =inner_join_df.rename(columns={0:'energy_supplied'})inner_join_df =inner_join_df.rename(columns={year:'energy_emissions'})inner_join_df['emission_per_energy'] = inner_join_df['energy_emissions']/inner_join_df['energy_supplied']plt.figure(figsize=(20,10))  # Adjust the size as neededinner_join_df = inner_join_df.sort_values(by='emission_per_energy',ascending=False)inner_join_df_plt = inner_join_df[0:20]ax=sns.barplot(x=inner_join_df_plt.mrio_code,y=inner_join_df_plt['emission_per_energy'], data=inner_join_df_plt)plt.xticks(rotation=90)plt.xlabel('Economy')plt.ylabel('Emission per Energy Unit Supplied')plt.title(str(year))for p in ax.patches:    ax.annotate(f'{p.get_height():.6f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90, weight='bold')os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")plt.savefig("energyemissionperenergy"+str(year)+"_prod.png")# MRIO Raw Data mriot_year = pd.read_excel("/Users/divyasangaraju/Library/CloudStorage/OneDrive-SharedLibraries-AsianDevelopmentBank/SDIU - GVC/Climate Change/IO Publication 2024/EE-MRIOTs_as of March 2024/ADB-EE-MRIO-"+str(year)+".xlsx")[4:]del mriot_year[mriot_year.columns[0]]del mriot_year[mriot_year.columns[0]]mriot_year = mriot_year.rename(columns={mriot_year.columns[0]: 's'})mriot_year = mriot_year.rename(columns={mriot_year.columns[1]: 's_sector'})mriot_year_trans = mriot_year.Ttest =mriot_year_trans.groupby(4).sum().Ttest['s'] = mriot_year_trans.iloc[1]test_filt = test[test['s'] =='c17']sum_result = pd.DataFrame(test_filt.sum(axis=0))# Largest Energy Consumer plt.figure(figsize=(20,10))  # Adjust the size as neededsum_result[0] = pd.to_numeric(sum_result[0], errors='coerce')# Drop rows where 'A' column contains non-numeric values (NaN)result = sum_result.dropna(subset=[0])result = sum_result.sort_values(by=0,ascending=False)result_plt =result[0:20] ax=sns.barplot(x=result_plt.index,y=result_plt[0], data=result_plt)plt.xticks(rotation=90)plt.xlabel('Economy')plt.ylabel('Energy Consumption Value (in Million)')plt.title("Energy Consumer("+ str(year)+")")for p in ax.patches:    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90, weight='bold')os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")# Save the plot to a fileplt.savefig("consumer"+str(year)+".png")# Close the plotplt.close()### Energy consumption per unit of Output produced in economy mriotyear_y = mriot_year[1:2569]output_sum = pd.DataFrame(mriotyear_y.sum(axis=1))output_sum = output_sum.rename(columns={output_sum.columns[0]: 'output'})output_sum=output_sum.reset_index()output_sum = output_sum.rename(columns={output_sum.columns[0]: 'mrio_code'})result = result.rename(columns={0: 'e_consumption'})result=result.reset_index()result = result.rename(columns={4: 'mrio_code'})result = result[0:72]result = result[['mrio_code','e_consumption']]output_sum['output'] = output_sum['output'].astype(float)result['e_consumption'] = result['e_consumption'].astype(float)output_consump = pd.merge(output_sum, result, on='mrio_code', how='inner')output_consump['energyconsump_per_output'] = output_consump['e_consumption']/output_consump['output']plt.figure(figsize=(20,10))  # Adjust the size as neededoutput_consump = output_consump.sort_values(by='energyconsump_per_output',ascending=False)output_consump_plt =output_consump[0:20] ax=sns.barplot(x=output_consump_plt.mrio_code,y=output_consump_plt['energyconsump_per_output'], data=output_consump_plt)plt.xticks(rotation=90)plt.xlabel('Economy')plt.ylabel('Energy Consumption per Unit of Output')plt.title(year)for p in ax.patches:    ax.annotate(f'{p.get_height():.1f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 25), textcoords='offset points', rotation=90, weight='bold')os.chdir("/Users/divyasangaraju/Documents/Work/ADB/IO Publication/charts")# Save the plot to a fileplt.savefig("energyconsump_output"+str(year)+".png")plt.show()# Close the plotplt.close()### Emissions produced per unit of Consumption plt.clf()eby_energy = eby[eby['sector']=='Energy']eby_energy['emissions'] = pd.to_numeric(eby_energy['emissions'])eby_energy['emissions_inmil'] = eby_energy['emissions']/1000000e_consumer = pd.DataFrame(eby_energy.groupby(['r','t'])['emissions_inmil'].sum()).sort_values(by='emissions_inmil',ascending=False).reset_index()e_consumer=e_consumer.pivot(index='r', columns='t', values='emissions_inmil').reset_index()e_consumer=e_consumer.rename(columns={'r':'mrio'})country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')e_consumer_cname = pd.merge(e_consumer, country_dict, on='mrio', how='inner')################################################################################################################################################################################# Bilateral Linkage Chartimport pandas as pdimport holoviews as hvfrom holoviews import opts, dimyear = 2021hv.extension('matplotlib')hv.output(fig='svg', size=250)eby_energy= eby[eby['sector']=='Energy']eby_energy['emissions_inmil'] = eby_energy['emissions']/1000000e_producer_consumer = pd.DataFrame(eby_energy.groupby(['r','s','t'])['emissions_inmil'].sum()).sort_values(by='emissions_inmil',ascending=False).reset_index()e_producer_consumer=e_producer_consumer.dropna()e_producer_consumer = e_producer_consumer[e_producer_consumer.t ==year]e_producer_consumer=e_producer_consumer.rename(columns={'s':'source','r':'target','emissions_inmil':'value','t':'node'})e_producer_consumer = e_producer_consumer[['target','source','value']]country_dict = pd.read_csv('/Users/divyasangaraju/Documents/Work/ADB/IO Publication/RawData/countries.csv', encoding='latin1')country_dict.set_index('mrio')node = country_dict[['mrio_code','mrio']]node=node.rename(columns={'mrio_code':'name','mrio':'index'})nodes = hv.Dataset(node.reset_index(), 'index')e_producer_consumer['value'].fillna(0, inplace=True)#customization of chartimg = hv.Chord((e_producer_consumer, nodes)).select(value=(0.3, None)).opts(    opts.Chord(fontsize={'title': 25, 'labels': 15, 'xticks': 20, 'yticks': 20},title="Bilateral Emission Flows " + str(year),cmap='Category10', node_size=10,sublabel_size=35, cbar_width=5,edge_cmap='Category10', edge_color=dim('source').astype(str), labels='name', node_color=dim('index').astype(str)))hv.render(img)