# hw1_094295

## Explanation for all the files:
1. create.py --> creates the 'all_new_data.csv' file that contains all the rows from the train files, not including the irrelevent rows as explained in the insturcions. 
2. create_test.py --> same as 1. but for the test data, creates 'new_test_data_all_cols.csv'.
3. agg_train.py --> creates the aggregated train data, as explained in the report, and creates the scv 'agg_last_data_new.csv'.
4. agg_test.py --> creates the aggregated test data, as explained in the report, and creates the scv 'agg_last_data_test_new.csv'.
5. train.py --> our final train of the model, saves the pkl file of the final result - as adaboost_trained_model.
6. train_randomforest.py --> training of another model using random forest, daves the model as rf_trained_model.
7. train_xgb --> training of another model using xgboost, daves the model as xgb_trained_model.
8. predict.py --> predicts the sepsis label and creates 'prediction.csv'.
9. files that were used during the analysis and computation of f1 score - importance.py, analize_different_groups_by_age.py, analize_different_groups_by_gender.py,   
   analize_different_groups_by_iculus.py, analize_different_groups_by_temp.py, evaluate_randomforest.py, final_exploration.py.
   
### Please notice that the folders - '05May23' and 'adaboost_last_row_relevant_cols' are our beckup folders, please ignore them. 

In order to run predict.py- 
1. Run conda env create --file environment.yml
2. Run conda activate hw1_env
3. Run python predict.py 'path to test/ unseen data'. Note that the current prediction.csv file in our repo is the predictions for the rest data. 
