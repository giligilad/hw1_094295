import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv('all_new_data.csv', index_col=0)
dict1= dict()
for col in data.columns:
    null_val = data[col].isna().sum()
    num_rows = len(data)
    value= (null_val/num_rows)*100
    dict1[col]=[value]


for col in data.columns:
    data[col] = data[col].fillna(data.groupby('id')[col].transform('mean'))
for col in data.columns:
    null_val = data[col].isna().sum()
    num_rows = len(data)
    value= (null_val/num_rows)*100
    dict1[col].append(value)
print("end2")

relevant_columns = list()
for col in data.columns:
    null_val = data[col].isna().sum()
    num_rows = len(data)
    value= (null_val/num_rows)*100
    if value < 20:
        relevant_columns.append(str(col))

data = data.filter(items=relevant_columns)
for col in data.columns:
  data[col] = data[col].fillna(data[col].mean())

print(len(data.columns))


for col in data.columns:
    null_val = data[col].isna().sum()
    num_rows = len(data)
    value= (null_val/num_rows)*100
    dict1[col].append(value)
print("end3")

data = data[relevant_columns]
agg_data = data.groupby('id').tail(1)
print(type(agg_data))
agg_data.reset_index(inplace=True)
print(agg_data)
for i in range(len(agg_data)):
    agg_data.loc[i,'SepsisLabel'] = 1 if agg_data.loc[i,'SepsisLabel'] > 0 else 0

#agg_data = data.groupby(['id'])[relevant_columns].mean()
#agg_data= data.groupby(['id'])[relevant_columns].last()

#agg_data['SepsisLabel'] = agg_data['SepsisLabel'].apply(lambda x: 1 if x > 0 else x)
print(len(agg_data))
print(agg_data.head())
df_sepsis = agg_data[agg_data['SepsisLabel']> 0]
print(len(df_sepsis))
df_no_sepsis = agg_data[agg_data['SepsisLabel']==0.0]
print(len(df_no_sepsis))

#
# # agg_data.drop_duplicates(inplace=True)
# # agg_data.reset_index(inplace=True)
# #
agg_data.to_csv("agg_last_data_new.csv", index=False)