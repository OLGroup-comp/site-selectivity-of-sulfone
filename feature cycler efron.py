import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
import openpyxl
import xlsxwriter
from openpyxl import load_workbook
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import cross_val_score
from numpy import absolute, mean

def efron_rsquare(y, y_pred):
    n = float(len(y))
    t1 = np.sum(np.power(y - y_pred, 2.0))
    t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
    return 1.0 - (t1 / t2)

def random_forest(X, y_train):
    np.random.seed(777)
    # redwood = BernoulliNB()
    redwood = LogisticRegression()
    redwood.fit(X, y_train)

    # cooralation coefficients
    results = []

    y_t = redwood.predict_proba(X)
    # print(y_t)
    y_prob_t = y_t[:, 1]
    ef_t = efron_rsquare(y_train, y_prob_t)

    results.append(ef_t)

    return results


train_df = pd.read_excel(r"C:\Users\trevi\Downloads\BernoulliNB\BernoulliNB\dataset.xlsx", sheet_name='Train ds_add', header=None)

full_table = train_df.iloc[0:, 2:272]
y = train_df.iloc[0:, 1]
model_column_list = []

for m in range(1, 20):
    #function call

    model_vals_by_column = []

    column_list = list(range(2, 271))
    for j in model_column_list:
        column_list.remove(j)
    # print(column_list)
    for i in column_list:
        # print(i)
        model_column_list_trial = []
        for j in model_column_list:
            model_column_list_trial.append(j)
        model_column_list_trial.append(i)
        # print(model_column_list_trial)

        data_untransposed = []
        for column in model_column_list_trial:
            # print(column)
            data_untransposed.append(full_table[column].tolist())
        np.random.seed(777)
        data_untransposed = np.array(data_untransposed).transpose()
        X = []
        X_test = []
        y_train = []
        y_test = []
        for o in range(len(data_untransposed)):
            if o not in []:
                X.append(data_untransposed[o])
            else:
                X_test.append(data_untransposed[o])
        for p in range(len(y)):
            if p not in []:
                y_train.append(y[p])
            else:
                y_test.append(y[p])

        trial_results = random_forest(X, y_train)
        entry = []
        entry.append(i)
        for l in trial_results:
           entry.append(l)
        model_vals_by_column.append(entry)
  
    best_idx = 0
    best_idx_excel = model_vals_by_column[0][0]
    while model_vals_by_column[best_idx][0] in model_column_list:
        best_idx = best_idx + 1
        best_idx_excel = model_vals_by_column[best_idx][0]
    # print(len(model_vals_by_column))
    for k in range(len(model_vals_by_column)):
        if model_vals_by_column[k][1] >= model_vals_by_column[best_idx][1] and model_vals_by_column[k][0] not in model_column_list:
            best_idx = k
            best_idx_excel = model_vals_by_column[k][0]

    model_column_list.append(best_idx_excel)
    print(str(model_vals_by_column[best_idx][1]) + ";" + str(model_column_list))
