from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd
import os

# read the aggregate csv:
combined_df = pd.read_csv('unseen_data.csv', index_col=0)
print(combined_df.columns)



all_columns = ['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP', 'HospAdmTime',
             'Creatinine', 'SBP', 'Glucose', 'Magnesium', 'Gender', 'O2Sat', 'Potassium', 'Age','id','SepsisLabel']

# all_columns_for_xgb = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
#              'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS','id','SepsisLabel']

# all_columns = all_columns_for_xgb
x_columns = all_columns[:-1]

combined_df = combined_df[all_columns]

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df.groupby('id')[col].transform('mean'))

for col in combined_df.columns:
    combined_df[col] = combined_df[col].fillna(combined_df[col].mean())


agg_data = combined_df.groupby('id').tail(1)
agg_data_old = agg_data[agg_data['Age'] > 50]
agg_data_young= agg_data[agg_data['Age'] <= 50]

agg_data_old.reset_index(inplace=True)
agg_data_young.reset_index(inplace=True)
print("len of old:" , len(agg_data_old))
print("len of young:" , len(agg_data_young))

for i in range(len(agg_data_old)):
    agg_data_old.loc[i, 'SepsisLabel'] = 1 if agg_data_old.loc[i, 'SepsisLabel'] > 0 else 0

for i in range(len(agg_data_young)):
    agg_data_young.loc[i, 'SepsisLabel'] = 1 if agg_data_young.loc[i, 'SepsisLabel'] > 0 else 0

print("for young:")
print("not sick",len(agg_data_young[agg_data_young['SepsisLabel'] == 0]))
print("sick",len(agg_data_young[agg_data_young['SepsisLabel'] == 1]))

print("for old:")
print("not sick",len(agg_data_old[agg_data_old['SepsisLabel'] == 0]))
print("sick",len(agg_data_old[agg_data_old['SepsisLabel'] == 1]))

X_test_old = agg_data_old[x_columns[:-1]]
y_test_old = agg_data_old['SepsisLabel']

X_test_young = agg_data_young[x_columns[:-1]]
y_test_young = agg_data_young['SepsisLabel']


# load the model:
#filename = 'adaboost_trained_model'
filename = 'rf_trained_model'
#filename = 'xgb_trained_model'

loaded_model = pickle.load(open(filename, 'rb'))

#for adaboost and random forest:
#calculate f1 score- old:
from sklearn.metrics import f1_score
y_pred= loaded_model.predict(X_test_old)
print("number of predicted sick- old: ", len(y_pred[y_pred==1]))
f1 = f1_score(y_test_old, y_pred)
print("F1-score for old: {:.2f}".format(f1))

#calculate f1 score- young:
from sklearn.metrics import f1_score
y_pred= loaded_model.predict(X_test_young)
print("number of predicted sick- young: ", len(y_pred[y_pred==1]))
f1 = f1_score(y_test_young, y_pred)
print("F1-score for young: {:.2f}".format(f1))

#for xgb:
# import xgboost as xgb
#
# #calculate f1 score- old:
# from sklearn.metrics import f1_score
# dtest = xgb.DMatrix(X_test_old, label=y_test_old)
# y_pred = loaded_model.predict(dtest)
# y_pred = [int(round(x)) for x in y_pred]
# print("number of predicted sick- old: ", len([x for x in y_pred if x ==1]))
# f1 = f1_score(y_test_old, y_pred)
# print("F1-score: {:.2f}".format(f1))
#
# #calculate f1 score- young:
# from sklearn.metrics import f1_score
# dtest = xgb.DMatrix(X_test_young, label=y_test_young)
# y_pred = loaded_model.predict(dtest)
# y_pred = [int(round(x)) for x in y_pred]
# print("number of predicted sick- young: ", len([x for x in y_pred if x ==1]))
# f1 = f1_score(y_test_young, y_pred)
# print("F1-score: {:.2f}".format(f1))