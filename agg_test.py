import pandas as pd

data = pd.read_csv('new_test_data.csv', index_col=0)
all_cols = ['O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium',
                                                        'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb',
                                                        'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS',
                                                        'SepsisLabel', 'HR', 'id']
for col in data.columns:
    data[col] = data[col].fillna(data.groupby('id')[col].transform('mean'))

for col in data.columns:
    data[col] = data[col].fillna(data[col].mean())

#agg_data = data.groupby(['id'])[all_cols].mean()
print(type(data))
agg_data = data.groupby('id').tail(1)
print(type(agg_data))
agg_data.reset_index(inplace=True)
print(agg_data)
for i in range(len(agg_data)):
    agg_data.loc[i,'SepsisLabel'] = 1 if agg_data.loc[i,'SepsisLabel'] > 0 else 0
#.apply(lambda x: 1 if x > 0 else x)

print(len(agg_data[agg_data['SepsisLabel']==0]))
print(len(agg_data[agg_data['SepsisLabel'] ==1]))

agg_data.to_csv("agg_last_data_test_new.csv", index=False)
