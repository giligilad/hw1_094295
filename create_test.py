import os
import pandas as pd

train_directory = 'data/test/'
all_train_files = [file for file in os.listdir(train_directory) if file.endswith('.psv')]
combined_df = pd.DataFrame()
for i, file in enumerate(all_train_files):
    if i%1000 ==0:
        print(i)
    file_path = train_directory + file
    # df = pd.read_csv(file_path, delimiter='|', usecols=['O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'BUN', 'Calcium',
    #                                                     'Creatinine', 'Glucose', 'Magnesium', 'Potassium', 'Hct', 'Hgb',
    #                                                     'WBC', 'Platelets', 'Age', 'Gender', 'HospAdmTime', 'ICULOS',
    #                                                     'SepsisLabel', 'HR'])
    df = pd.read_csv(file_path, delimiter='|')
    id = file_path.split("patient_")[-1].strip(".psv")
    df['id'] = [id] * len(df)
    df['label_change'] = df.SepsisLabel.diff().fillna(0) != 0
    if (df['label_change'].sum() == 2):
        shift_index = df.loc[df['label_change'].idxmax(), :].name
        df = df.loc[:shift_index, :]
    elif (df['SepsisLabel'][0] == 1):
        df = df.iloc[[0], :]
    if i == 0:
        combined_df = df
    else:
        combined_df = pd.concat([combined_df, df], ignore_index=True)

combined_df.reset_index(inplace=True)
print(combined_df.columns)
combined_df.to_csv("new_test_data_all_cols.csv", index=False)