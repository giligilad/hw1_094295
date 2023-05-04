import os
import pandas as pd

train_directory = 'data/train/'
all_train_files = [file for file in os.listdir(train_directory) if file.endswith('.psv')]
final_df = pd.DataFrame()
for i, file in enumerate(all_train_files):
    if i%1000 ==0:
        print(i)
    file_path = train_directory + file
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
        final_df = df
    else:
        final_df = pd.concat([final_df, df], ignore_index=True)

final_df.reset_index(inplace=True)
final_df.to_csv("all_new_data.csv", index=False)