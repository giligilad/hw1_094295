import pickle
import pandas as pd

combined_df = pd.read_csv('agg_last_data_test_new.csv', index_col=0)
all_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
             'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS','id','SepsisLabel']
x_columns = all_columns[:-1]

combined_df = combined_df[all_columns]

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df.groupby('id')[col].transform('mean'))

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mean())

agg_data = combined_df.groupby('id').tail(1)
print(agg_data)
agg_data.reset_index(inplace=True)
print(agg_data)

for i in range(len(agg_data)):
    agg_data.loc[i, 'SepsisLabel'] = 1 if agg_data.loc[i, 'SepsisLabel'] > 0 else 0

print(len(agg_data[agg_data['SepsisLabel'] == 0]))
print(len(agg_data[agg_data['SepsisLabel'] == 1]))

X_test = agg_data[x_columns[:-1]]
print(X_test)
y_test = agg_data['SepsisLabel']

# load the model:
filename = 'xgb_trained_model'
loaded_model = pickle.load(open(filename, 'rb'))

#calculate f1 score:
import xgboost as xgb

from sklearn.metrics import f1_score
dtest = xgb.DMatrix(X_test, label=y_test)
y_pred = loaded_model.predict(dtest)
y_pred = [int(round(x)) for x in y_pred]
f1 = f1_score(y_test, y_pred)
print("F1-score: {:.2f}".format(f1))
