import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import sklearn

from sklearn.model_selection import cross_val_predict, KFold

data = pd.read_csv('agg_last_data_new.csv')
all_columns = list(data.columns)
print(all_columns)

# drop gender,HospAdmTime,ICULOS,Index,Id
x_columns = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium', 'Creatinine', 'Glucose',
             'Magnesium', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS']
print(x_columns)

X_train = data[x_columns]
# print(X_train)
y_train = data['SepsisLabel']
# print(y_train)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import f_regression

fc = SelectKBest(f_classif, k=20)
X_new = fc.fit_transform(X_train, y_train)
top_features = sorted(zip(x_columns, fc.scores_), key=lambda x: x[1], reverse=True)[:20]
chosen = []
for feature in top_features:
    chosen.append(feature[0])
print(chosen)

X_train = data[chosen]
test_data = pd.read_csv('agg_last_data_test_new.csv')
X_test = test_data[chosen]
y_test = test_data['SepsisLabel']

# normlize df:- didnt work with random forest
from sklearn import preprocessing

# xtrain = X_train.values
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(xtrain)
# X_train = pd.DataFrame(x_scaled)
#
# xtest = X_test.values
# min_max_scaler = preprocessing.MinMaxScaler()
# x_scaled = min_max_scaler.fit_transform(xtest)
# X_test = pd.DataFrame(x_scaled)

# random forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train,y_train)
#y_pred = rf.predict(X_test)

# adaboost:
from sklearn.ensemble import AdaBoostClassifier

# clf = AdaBoostClassifier(n_estimators=100, random_state=0)
# clf.fit(X_train, y_train)
#y_pred = clf.predict(X_test)

# #mlp
# from sklearn.neural_network import MLPClassifier
# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# y_pred= clf.predict(X_test)

# #logistic regression
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# y_pred= clf.predict(X_test)

# KNN
# from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=5)
# neigh.fit(X_train, y_train)
# y_pred= neigh.predict(X_test)


# Calculate the F1-score
#save the model:
import pickle
filename='rf_trained_model'
pickle.dump(rf,open(filename,'wb'))

#f1 = f1_score(y_test, y_pred)

#print("F1-score: {:.2f}".format(f1))
