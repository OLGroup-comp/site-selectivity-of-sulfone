import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)

def random_forest(X_train, X_test, y_train):

    column_converter = ['Δ(c1-c4) NBO','Δ(c1-c4) LUMO', 'r(c1/c3) LUMO', 'r(c4/c2) LUMO', 'c1xc3 HOMO', 
                        'c2xc4 HOMO', 'c1xc2 LUMO', 'c3xc4 LUMO', 'Δ([C1-S]-[C4-S]) Dist', 'Δ(C1-C4) LUMO+1', 'Δ(c2-c3) Fukui']

    # print(X_train)
    #-----convert column_list to names
    column_list = []
    #----call regression function
    svr = BernoulliNB()

    #-----fit model to training data
    svr.fit(X_train, y_train)

    #-----predict data values
    y_predicted_train =svr.predict(X_train)
    # y_predicted_test = svr.predict(X_test)


    #-----transform data for csv output
    df_train = []
    df_test = []




    # df_test.append(y_test)
    # df_test.append(y_predicted_test)
    # df_test = pd.DataFrame(np.array(df_test).transpose())
    # df_test.to_csv(r"C:\Users\trevi\Downloads\Test Prediction.csv",
    #                 index=False, header=["Experimental ", " Predicted"])



    #-----feature importance
    # perform permutation importance
    # results = permutation_importance(svr, X_train, y_train, scoring='neg_mean_squared_error')
    results = svr.feature_log_prob_
    importance = abs(results[0])
    print(importance)
    # importance = results.importances_mean


    print("Importance: " + str(len(importance)) + "Column_converter: " + str(len(column_converter)))
    df = pd.DataFrame({
         'col1': column_converter,
         'col2': importance
    })
    # print(df)

    df_values = []
    df_values.append(df['col1'].to_list())
    df_values.append(df['col2'].to_list())
    df_values = pd.DataFrame(np.array(df_values).transpose())
    df_values.to_csv(r"C:\Users\trevi\Downloads\BernoulliNB\BernoulliNB\Feature_importance_Thiophene.csv", index=False)

    #-----extra data section

    results = []
 
    #column_converter importance

    # print(str(results[0]) + ";" + str(results[1]) + ";" + str(results[2]) + ";" + str(results[3]))

    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    column_converter_sorted = []
    importance_sorted = []
    importance, column_converter = zip(*sorted(zip(importance, column_converter)))
    print(np.array(importance).transpose())
    for e in column_converter:
        print(e + ";")
 
    return results

#------------------------------------------
column_list = [34, 46, 102, 110, 135, 138, 140, 145, 226, 231, 244]
row_test_list = []

#-----internal data grab
train_df = pd.read_excel(r'C:\Users\trevi\Downloads\BernoulliNB\BernoulliNB\dataset.xlsx', 
                        sheet_name='Train ds_add', header=None)
# full_table = train_df.iloc[0:, 2:260]
full_table = train_df.iloc[:, column_list]
y = train_df.iloc[0:, 1]
# 
np.random.seed(4746)

X_train = []
X_test = []
y_train = []
y_test = []
for o in full_table:
    train_adder = []
    test_adder = []
    for w in range(len(full_table[o])):
        if w not in row_test_list:
            train_adder.append(full_table[o][w])
        else:
            test_adder.append(full_table[o][w])
    X_train.append(train_adder)
    X_test.append(test_adder)
X_train = np.array(X_train).transpose()
X_test = np.array(X_test).transpose()
for p in range(len(y)):
    if p not in row_test_list:
        y_train.append(y[p])
    else:
        y_test.append(y[p])

trial_results = random_forest(X_train, X_test, y_train)
print(trial_results)