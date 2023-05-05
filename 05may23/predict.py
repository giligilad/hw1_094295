from sklearn.ensemble import AdaBoostClassifier
import pickle
import pandas as pd
import os

# load data:
# train_directory = 'data/test/'
# all_train_files = [file for file in os.listdir(train_directory) if file.endswith('.psv')]
# combined_df = pd.DataFrame()
# for i, file in enumerate(all_train_files):
#     if i % 1000 == 0:
#         print(i)
#     file_path = train_directory + file
#     df = pd.read_csv(file_path, delimiter='|')
#     id = file_path.split("patient_")[-1].strip(".psv")
#     df['id'] = [id] * len(df)
#     df['label_change'] = df.SepsisLabel.diff().fillna(0) != 0
#     if (df['label_change'].sum() == 2):
#         shift_index = df.loc[df['label_change'].idxmax(), :].name
#         df = df.loc[:shift_index, :]
#     elif (df['SepsisLabel'][0] == 1):
#         df = df.iloc[[0], :]
#     if i == 0:
#         combined_df = df
#     else:
#         combined_df = pd.concat([combined_df, df], ignore_index=True)
# combined_df.reset_index(inplace=True)
# print(combined_df.columns)
#combined_df.to_csv("new_test_data_all_cols.csv", index=False)


# read the aggregate csv:
combined_df = pd.read_csv('agg_last_data_test_new.csv', index_col=0)
all_columns = ['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP', 'HospAdmTime',
             'Creatinine', 'SBP', 'Glucose', 'Magnesium', 'Gender', 'O2Sat', 'Potassium', 'Age','id','SepsisLabel']
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
y_test = agg_data['SepsisLabel']

# load the model:
filename = 'adaboost_trained_model'
loaded_model = pickle.load(open(filename, 'rb'))

#calculate f1 score:
from sklearn.metrics import f1_score
y_pred= loaded_model.predict(X_test)
f1 = f1_score(y_test, y_pred)
print("F1-score: {:.2f}".format(f1))

agg_data['id']= agg_data['id'].apply(lambda x:'patient_' + str(x))

#create labeled csv:
dict_labeled = {'id' : agg_data['id'], 'prediction' : y_pred}
final_df = pd.DataFrame(dict_labeled)
final_df.to_csv('prediction.csv',index = False)