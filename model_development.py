import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.semi_supervised import LabelPropagation
from xgboost.sklearn import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import RadiusNeighborsClassifier, KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import patsy
import openpyxl
import xlsxwriter
from openpyxl import load_workbook


def log_regression(x_values, excludeRows, read_file_path, write_file_path, output_file_name, write_line, write_row):
    # File naming scheme
    file_name = output_file_name
    for col, value in enumerate(x_values):
        file_name = file_name + '_' + str(x_values[col]) 
    for row, value in enumerate(excludeRows):
        file_name = file_name + str(excludeRows[row])

    # print("-----")
    # print(excludeRows)
    # print("-----")

    # Load the training data from Excel file
    train_df = pd.read_excel(f"{read_file_path}", sheet_name='train ds_add_092223', header=None)         # train ds_add_092223 or train ds_add or train ds_update 0.5 092523
    # train_df = pd.read_excel(f"{read_file_path}", sheet_name='train 1kcal', header=None)         # train ds_add_092223 or train 1kcal or train ds_update 0.5 092523
    # Record rows to delete
    delete_rows = train_df.iloc[:, 0].values
    exclude_rw_idx = []
    for i in excludeRows:
        counter = 0
        for j in delete_rows:
            if j == i:
                exclude_rw_idx.append(counter)
            counter = counter + 1
    idx = [i for i, _ in enumerate(delete_rows)]
    for i in exclude_rw_idx:
        idx.remove(i)
    # Read data and remove desired rows
    x_d_train = []
    for column in x_values:
        x_d_train.append(train_df.iloc[idx, column].values)
    X_train = np.array(x_d_train).transpose()
    y_train = train_df.iloc[idx, 1].values

    # perform VIF test (place right between X_train and y_train definitions)
    vif_dataframe = pd.DataFrame(X_train)

    def vif_scores(df):
        VIF_Scores = pd.DataFrame()
        VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return VIF_Scores

        # Load the validation data from Excel file

    # val_df = pd.read_excel(f"{read_file_path}", sheet_name='Validation', header=None)              #---------------------------------------------change sheet name
    val_df = pd.read_excel(f"{read_file_path}", sheet_name='Validation-all', header=None)  
    # Split the validation data into X and y
    x_d_val = []
    for column in x_values:
        x_d_val.append(val_df.iloc[:, column].values)

    X_val = np.array(x_d_val).transpose()
    y_val = val_df.iloc[:, 1].values
    row_val = val_df.iloc[:, 0].values

    np.random.seed(777)

    ## Logisitic Regression
    ## Training Set
    # Create a logistic regression model
    #model_t = LogisticRegression(penalty='none')
    # model_t = SVC(kernel ='rbf', random_state=777, probability=True)
    # model_t = QuadraticDiscriminantAnalysis()
    # model_t = RandomForestClassifier(random_state=777)
    # model_t = RandomForestClassifier(random_state=777, n_estimators=1400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10, bootstrap=False)
    #model_t = GradientBoostingClassifier(random_state=777)
    #model_t = XGBClassifier(random_state=777)
    #model_t = DecisionTreeClassifier(random_state=777)
    # model_t = RadiusNeighborsClassifier(radius=100.0)
    model_t = BernoulliNB()
    # model_t = GaussianNB()
    # model_t = CalibratedClassifierCV()

    # Fit the model to the training data
    model_t.fit(X_train, y_train)

    # Extract coefficient(s) of indepdent variable(s)
    #w_t = np.array(model_t.coef0).transpose()

    # Predict the outcome for the training set
    y_pred_t = model_t.predict(X_train)

    # Probability of y for training
    y_t = model_t.predict_proba(X_train)
    y_prob_t = y_t[:, 1]
    # class_labels = model_t.predict(X_train)
    # # y_prob_ex = y_ex[:, 1]
    # failures = []
    # failure_status = []
    # for i in range(len(class_labels)):
    #     if y_t[i] != class_labels[i]:
    #         failures.append(row_val[i])
    #         if y_t[i] == 0:
    #             failure_status.append("False Positive")
    #         else:
    #             failure_status.append("False Negative")

    np.random.seed(777)
    ## Validation Set
    # Create a logistic regression model
    # model_v = LogisticRegression(penalty='none')
    # model_v = SVC(kernel = 'rbf', random_state=777, probability=True)
    # model_v = QuadraticDiscriminantAnalysis()
    # model_v = RandomForestClassifier(random_state=777)
    # model_v = RandomForestClassifier(random_state=777, n_estimators=1400, min_samples_split=2, min_samples_leaf=1, max_features='sqrt', max_depth=10, bootstrap=False)
    #model_v = GradientBoostingClassifier(random_state=777)
    #model_v = XGBClassifier(random_state=777)
    #model_v = DecisionTreeClassifier(random_state=777)
    # model_v = RadiusNeighborsClassifier(radius=100.0)
    model_v = BernoulliNB()
    # model_v = GaussianNB()
    # model_v = CalibratedClassifierCV()

    # Fit the model to the validation data
    model_v.fit(X_val, y_val)

    # Extract coefficient(s) of indepdent variable(s)
    #w_v = np.array(model_v.coef0).transpose()

    # Predict the outcome for the validation set
    y_pred_v = model_v.predict(X_val)

    # Probability of y for validation
    y_v = model_v.predict_proba(X_val)
    y_prob_v = y_v[:, 1]

    ## External Validation
    # predict the ouctome and probability
    y_pred_ex = model_t.predict(X_val)
    y_ex = model_t.predict_proba(X_val)
    class_labels = model_t.predict(X_val)
    y_prob_ex = y_ex[:, 1]
    failures = []
    failure_status = []
    for i in range(len(class_labels)):
        if y_val[i] != class_labels[i]:
            failures.append(row_val[i])
            if y_val[i] == 0:
                failure_status.append("False Positive external")
            else:
                failure_status.append("False Negative external")
    # print(y_val)
    # print("-----")
    # print(failures)

    # extra commands
    def efron_rsquare(y, y_pred):
        n = float(len(y))
        t1 = np.sum(np.power(y - y_pred, 2.0))
        t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
        return 1.0 - (t1 / t2)

    ## Statsmodel logistic regression
    #log_reg_v = sm.Logit(y_val, X_val)
    #results_v = log_reg_v.fit()

    #log_reg_t = sm.Logit(y_train, X_train)
    #results_t = log_reg_t.fit()

    # print out logistic tests and significant data
    acc_v = accuracy_score(y_val, y_pred_v)
    f1_v = f1_score(y_val, y_pred_v)
    ef_v = efron_rsquare(y_val, y_prob_v)
    acc_t = accuracy_score(y_train, y_pred_t)
    f1_t = f1_score(y_train, y_pred_t)
    ef_t = efron_rsquare(y_train, y_prob_t)
    acc_ex = accuracy_score(y_val, y_pred_ex)
    f1_ex = f1_score(y_val, y_pred_ex)
    ef_ex = efron_rsquare(y_val, y_prob_ex)
    #true_mac_score_v = results_v.prsquared
    #true_mac_score_t = results_t.prsquared

    if f1_ex>0.01:                 #values to change from leave one out model f1_ex 0.8 ef_ex 0.76085 acc_ex 0.8
        ## CONFUSION MATRIX
        # confusion matrix generation training
        cm_t = confusion_matrix(y_train, y_pred_t)


        # confusion matrix generation validation
        cm_v = confusion_matrix(y_val, y_pred_v)



        # confusion matrix generation external validation
        cm_ex = confusion_matrix(y_val, y_pred_ex)


        d = {'col1': ["accuracy training", "f1 training", "efron training", \
                      "cf matrix", "accuracy training", "f1 training", "efron training", \
                      "cf matrix", "accuracy training", "f1 training", "efron training",\
                      "cf matrix", "File Name"],\
             'col2': ["training", "training", "training", "training", "validation", "validation", \
                                                                  "validation", "validation", "external",  "external",  \
                                                                  "external",  "external", " = "], \
            #  'col1': ["accuracy training", "f1 training", "efron training", \
                    #   "mcfadden training", "cf matrix", "accuracy training", "f1 training", "efron training", \
                    #   "mcfadden training", "cf matrix", "accuracy training", "f1 training", "efron training",\
                    #   "cf matrix", "File Name"],\
            #  'col2': ["training", "training", "training", "training", "training", "validation", "validation", "validation", \
                                                                #   "validation", "validation", "external",  "external",  \
                                                                #   "external",  "external", " = "], \
             "col3": [acc_t, f1_t, ef_t, cm_t, acc_v, f1_v, ef_v, cm_v, \
                      acc_ex, f1_ex, ef_ex, cm_ex, file_name]
            # "col3": [acc_t, f1_t, ef_t, true_mac_score_t, cm_t, acc_v, f1_v, ef_v, true_mac_score_v, cm_v, \
                    #  acc_ex, f1_ex, ef_ex, cm_ex, file_name]
             }
        results = {'col1': failures, \
                   'col2': failure_status
        }

        printout = pd.DataFrame(data=d)
        results_ad_on = pd.DataFrame(data=results)

        with pd.ExcelWriter("C:/Users/trevi/Desktop/python-codes/BernoulliNB/Bernoulli/output.xlsx",              #---------------------------------change path and name of output excel file
                         mode="a",
                         engine="openpyxl",
                         if_sheet_exists="overlay",
        ) as writer:
            start_row = write_row
            printout.to_excel(writer, sheet_name="BernNB ds_add", startcol=write_line, startrow=start_row)                #-----------change sheet names to write to
            results_ad_on.to_excel(writer, sheet_name="BernNB ds_add", startcol=write_line, startrow=start_row + 18)      #-----------change sheet names to write to
        #wb.save("C:/Users/Patrick/College/Computational Research/output.xlsx")
        return write_line + 6
    else:
        print(f"file {file_name} was tossed into the flame")
        return write_line



#--------------------------------------*****************FUNCTION CALLS*************************---------------------------------------------

# initialize starting row and column for excel writer
r=0
c=0

##echange 145 with 244
#246 is NUC
#[34, 44, 46, 102, 110, 135, 138, 140, 145, 225, 231]
#[34, 44, 46, 102, 110, 135, 138, 140, 244, 226, 231]

##this is for 1 kcal trianing set
# [226, 102, 231, 110, 219, 46, 212, 254, 205, 74, 210]

##this is for updated ds
# [226, 102, 110, 231, 49, 46, 244, 135, 138, 101, 111]
# BernoulliNB use this [226, 145, 140, 46, 231, 102, 110, 244, 135, 138, 34]
#---------------------------------------RUN SINGLE INDIVIDUAL TRIAL-------------------------------------
c = log_regression([226, 145, 140, 46, 231, 102, 110, 244, 135, 138, 34], [],
                  r'C:\Users\trevi\Desktop\python-codes\BernoulliNB\Bernoulli\logistic_regression_15_04_2025.xlsx',
                  r'C:/Users/trevi/Desktop/python-codes/BernoulliNB/Bernoulli/', 'Log-rerun', c, r)
