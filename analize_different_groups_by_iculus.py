from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd
import os

# read the aggregate csv:
combined_df = pd.read_csv('unseen_data.csv', index_col=0)

import numpy as np


all_columns = ['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP', 'HospAdmTime',
             'Creatinine', 'SBP', 'Glucose', 'Magnesium', 'Gender', 'O2Sat', 'Potassium', 'Age','id','SepsisLabel']

all_columns_for_xgb = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
             'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS','id','SepsisLabel']


all_columns = all_columns_for_xgb
x_columns = all_columns[:-1]

combined_df = combined_df[all_columns]

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df.groupby('id')[col].transform('mean'))

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mean())


agg_data = combined_df.groupby('id').tail(1)

print("o")
print(np.median(agg_data['ICULOS']))

agg_data_high = agg_data[agg_data['ICULOS'] > 39]
agg_data_low= agg_data[agg_data['ICULOS'] <= 39]


agg_data_high.reset_index(inplace=True)
agg_data_low.reset_index(inplace=True)
print("len of high ICULOS:" , len(agg_data_high))
print("len of low ICULOS:" , len(agg_data_low))

for i in range(len(agg_data_high)):
    agg_data_high.loc[i, 'SepsisLabel'] = 1 if agg_data_high.loc[i, 'SepsisLabel'] > 0 else 0

for i in range(len(agg_data_low)):
    agg_data_low.loc[i, 'SepsisLabel'] = 1 if agg_data_low.loc[i, 'SepsisLabel'] > 0 else 0

print("for high:")
print("not sick",len(agg_data_high[agg_data_high['SepsisLabel'] == 0]))
print("sick",len(agg_data_high[agg_data_high['SepsisLabel'] == 1]))

print("for Low:")
print("not sick",len(agg_data_low[agg_data_low['SepsisLabel'] == 0]))
print("sick",len(agg_data_low[agg_data_low['SepsisLabel'] == 1]))

X_test_high = agg_data_high[x_columns[:-1]]
y_test_high = agg_data_high['SepsisLabel']

X_test_low = agg_data_low[x_columns[:-1]]
y_test_low = agg_data_low['SepsisLabel']


# load the model:
#filename = 'adaboost_trained_model'
#filename = 'rf_trained_model'
filename = 'xgb_trained_model'
#
loaded_model = pickle.load(open(filename, 'rb'))
#
#for adaboost and random forest:
#calculate f1 score- high:
# from sklearn.metrics import f1_score
# y_pred= loaded_model.predict(X_test_high)
# print("number of predicted high: ", len(y_pred[y_pred==1]))
# f1 = f1_score(y_test_high, y_pred)
# print("F1-score for high: {:.2f}".format(f1))
#
# #calculate f1 score- low:
# from sklearn.metrics import f1_score
# y_pred= loaded_model.predict(X_test_low)
# print("number of predicted sick- low: ", len(y_pred[y_pred==1]))
# f1 = f1_score(y_test_low, y_pred)
# print("F1-score for low: {:.2f}".format(f1))

# #for xgb:
import xgboost as xgb

# #calculate f1 score- high temp:
from sklearn.metrics import f1_score
dtest = xgb.DMatrix(X_test_high, label=y_test_high)
y_pred = loaded_model.predict(dtest)
y_pred = [int(round(x)) for x in y_pred]
print("number of predicted sick- high temp: ", len([x for x in y_pred if x ==1]))
f1 = f1_score(y_test_high, y_pred)
print("F1-score: {:.2f}".format(f1))

#calculate f1 score- low temp:
from sklearn.metrics import f1_score
dtest = xgb.DMatrix(X_test_low, label=y_test_low)
y_pred = loaded_model.predict(dtest)
y_pred = [int(round(x)) for x in y_pred]
print("number of predicted sick- low temp: ", len([x for x in y_pred if x ==1]))
f1 = f1_score(y_test_low, y_pred)
print("F1-score: {:.2f}".format(f1))