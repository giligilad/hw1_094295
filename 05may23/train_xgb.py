import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn

from sklearn.model_selection import cross_val_predict, KFold


data = pd.read_csv('agg_last_data_new.csv')
all_columns= list(data.columns)
print(all_columns)

#drop gender,HospAdmTime,ICULOS,Index,Id
x_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
             'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS']
print(x_columns)

X_train = data[x_columns]
print(X_train)
y_train = data['SepsisLabel']
print(y_train)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression
# fc = SelectKBest(f_classif, k=5)
# X_new= fc.fit_transform(X_train, y_train)
# top_features = sorted(zip(x_columns, fc.scores_), key=lambda x: x[1], reverse=True)[:5]
# chosen= []
# for feature in top_features:
#     chosen.append(feature[0])
# print(chosen)

X_train = data[x_columns]

import xgboost as xgb
from sklearn.metrics import f1_score

# create DMatrix objects for X_train and X_test
dtrain = xgb.DMatrix(X_train, label=y_train)

# set XGBoost parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'error', # or 'logloss' or 'auc'
    'max_depth': 6,
    'eta': 0.3,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.5,
    'seed': 42
}

# train the model
num_rounds = 100
bst = xgb.train(params, dtrain, num_rounds)
#save the model:
import pickle
filename='xgb_trained_model'
pickle.dump(bst,open(filename,'wb'))

# make predictions on the test set
# y_pred = bst.predict(dtest)
# y_pred = [int(round(x)) for x in y_pred]
#
# # Calculate the F1-score
# f1 = f1_score(y_test, y_pred)
#
# print("F1-score: {:.2f}".format(f1))