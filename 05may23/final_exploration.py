import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv('agg_data_new.csv', index_col=0)
df = df.reset_index()

df_sepsis = df[df['SepsisLabel'] == 1]
print("df_sepsis", len(df_sepsis))
df_no_sepsis = df[df['SepsisLabel'] == 0.0]
print("df_no_sepsis", len(df_no_sepsis))

# all_cols =['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP',
# 'HospAdmTime', 'Creatinine', 'SBP', 'Glucose', 'Magnesium', 'Gender', 'O2Sat', 'Potassium', 'Age']
x = df_sepsis['Resp'].tolist()
y = df_no_sepsis['Resp'].to_list()
# bins = np.linspace(-10, 10, 100)
bins = np.linspace(0, 100, 20)
plt.xlabel("Resp")
plt.ylabel("Density")
plt.hist(x, bins, density=True, alpha=0.5, label='Sepsis')
plt.hist(y, bins, density=True, alpha=0.5, label='No sepsis')
plt.legend(loc='upper right')
plt.title('Resp histogram')
plt.savefig('Resp.png')
plt.show()
# Create a figure with subplots for each column
# fig, axs = plt.subplots(ncols=len(df.columns), figsize=(12, 4))


#################################
# corr matrix -
# all_features = ['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP', 'HospAdmTime', 'Creatinine',
#  'SBP', 'Glucose', 'Magnesium','SepsisLabel', 'Gender', 'O2Sat', 'Potassium', 'Age']

## ICULOS, BUN, HospAdmTime, O2Sat
### maybe -- WBC, Glucose
# mat_corr = df[all_features].corr()
#
# val = np.zeros_like(mat_corr)
# val[np.triu_indices_from(val)] = True
# plt.figure(figsize=(26,22))
# sns.heatmap(mat_corr, mask=val, square=True, annot=True, fmt=".2f", center=0, linewidths=.5)
# plt.title('Corr map for chosen features')
# plt.savefig('corr.png')
# plt.show()