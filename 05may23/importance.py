import pandas as pd
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt


#filename = 'adaboost_trained_model'
filename = 'rf_trained_model'
#filename = 'xgb_trained_model'
all_cols =['ICULOS', 'Temp', 'HR', 'WBC', 'BUN', 'Resp', 'Calcium', 'DBP', 'Hct', 'Hgb', 'MAP',
           'HospAdmTime', 'Creatinine', 'SBP', 'Glucose', 'Magnesium', 'Gender', 'O2Sat', 'Potassium', 'Age']

loaded_model = pickle.load(open(filename, 'rb'))

print(all_cols)
#xgb.plot_importance(loaded_model)
#plt.savefig('importance_xjboost.png')
importances = loaded_model.feature_importances_
print(importances)
