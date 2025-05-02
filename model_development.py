import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, matthews_corrcoef
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import shuffle
import patsy
import openpyxl
from openpyxl import load_workbook

def calc_mcfadden_R2(X_train, y_train, X_test, y_test):

    train_test_split = True
    if len(X_test) == 0:
        train_test_split = False
    lr = LogisticRegression().fit(X_train,y_train)
    # lr = BernoulliNB().fit(X_train, y_train)
    y_pred_prob = lr.predict_proba(X_train)[:,1] #List of predicted probabilities of output being "1" or above y-cut
    if train_test_split:
        y_pred_prob_test = lr.predict_proba(X_test)[:,1]
    null_probability = np.count_nonzero(y_train)/len(y_train) # overall probability of being "1" in train dataset
    if train_test_split:
        null_probability_test = np.count_nonzero(y_test)/len(y_test) # overall probability of being "1" in test dataset

    # Calculate log likelihood and null log likelihood
    null_log_likelihood = 0 #The sum of the log likelyhoods of the null

    for i in y_train:
        if i == 1:
            null_log_likelihood += np.log(null_probability)
        elif i == 0:
            null_log_likelihood += np.log(1-null_probability)
        else:
            print("ERROR!!!")

    log_likelihood = 0 #The sum of the log likelyhoods of the null

    for i in range(len(y_train)):
        if y_train[i] == 1:
            #calculate the log likelihood where likelihood = probability
            log_likelihood_i = np.log(y_pred_prob[i])
            log_likelihood += log_likelihood_i
        elif y_train[i] == 0:
            #calculate the log likelihood of 1-probability
            log_likelihood_i = np.log(1-y_pred_prob[i])
            log_likelihood += log_likelihood_i
        else:
            print("ERROR")

    Mcfadden_R2_test = 'N/A'
    if train_test_split:
        null_log_likelihood_test = 0 #The sum of the log likelyhoods of the null

        for i in y_test:
            if i == 1:
                null_log_likelihood_test += np.log(null_probability_test)
            elif i == 0:
                null_log_likelihood_test += np.log(1-null_probability_test)
            else:
                print("ERROR!!!")

        log_likelihood_test = 0 #The sum of the log likelyhoods of the null

        for i in range(len(y_test)):
            if y_test[i] == 1:
                #calculate the log likelihood where likelihood = probability
                log_likelihood_i = np.log(y_pred_prob_test[i])
                log_likelihood_test += log_likelihood_i
            elif y_test[i] == 0:
                #calculate the log likelihood of 1-probability
                log_likelihood_i = np.log(1-y_pred_prob_test[i])
                log_likelihood_test += log_likelihood_i
            else:
                print("ERROR")
        Mcfadden_R2_test = 1 - log_likelihood_test/null_log_likelihood_test

    Mcfadden_R2 = 1 - log_likelihood/null_log_likelihood
    return(Mcfadden_R2, Mcfadden_R2_test)

def log_regression(x_values, excludeRows, read_file_path, write_file_path, output_file_name, write_line, write_row):
    # File naming scheme
    file_name = output_file_name
    for col, value in enumerate(x_values):
        file_name = file_name + '_' + str(x_values[col]) 
    for row, value in enumerate(excludeRows):
        file_name = file_name + str(excludeRows[row])

    # Load the training data from Excel file
    train_df = pd.read_excel(f"{read_file_path}", sheet_name='Train ds_add', header=None)
    # train_df = pd.read_excel(f"{read_file_path}", sheet_name='Random Train', header=None)
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
    # X_train = shuffle(X_train, random_state=49) # This is for shuffle sets

    # perform VIF test (place right between X_train and y_train definitions)
    vif_dataframe = pd.DataFrame(X_train)

    def vif_scores(df):
        VIF_Scores = pd.DataFrame()
        VIF_Scores["VIF Scores"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return VIF_Scores

        # Load the validation data from Excel file

    val_df = pd.read_excel(f"{read_file_path}", sheet_name='Validation_set_1', header=None)
    # val_df = pd.read_excel(f"{read_file_path}", sheet_name='Random Valid', header=None)           
    # Split the validation data into X and y
    x_d_val = []
    for column in x_values:
        x_d_val.append(val_df.iloc[:, column].values)

    X_val = np.array(x_d_val).transpose()
    y_val = val_df.iloc[:, 1].values
    row_val = val_df.iloc[:, 0].values
    # X_val = shuffle(X_val, random_state=49) # This is for shuffle sets

    np.random.seed(777)
    ## Logisitic Regression
    ## Training Set
    model_t = LogisticRegression(penalty='none')
    # model_t = BernoulliNB()
    
    # Fit the model to the training data
    model_t.fit(X_train, y_train)

    # Predict the outcome for the training set
    y_pred_t = model_t.predict(X_train)

    # Probability of y for training
    y_t = model_t.predict_proba(X_train)
    y_prob_t = y_t[:, 1]

    np.random.seed(777)
    ## Validation Set
    model_v = LogisticRegression(penalty='none')
    # model_v = BernoulliNB()

    # Fit the model to the validation data
    model_v.fit(X_val, y_val)

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

    # extra commands
    def efron_rsquare(y, y_pred):
        n = float(len(y))
        t1 = np.sum(np.power(y - y_pred, 2.0))
        t2 = np.sum(np.power((y - (np.sum(y) / n)), 2.0))
        return 1.0 - (t1 / t2)

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
    mcfadden_R2, mcfadden_R2_test = calc_mcfadden_R2(X_train, y_train, X_val, y_val)
    print(f"McFadden R2: {mcfadden_R2}")
    print(f"McFadden R2 test: {mcfadden_R2_test}")

    if f1_ex>0.01:                
        ## CONFUSION MATRIX
        # confusion matrix generation training
        cm_t = confusion_matrix(y_train, y_pred_t)
        print('Training MCC')
        print(matthews_corrcoef(y_train, y_pred_t))

        # confusion matrix generation validation
        cm_v = confusion_matrix(y_val, y_pred_v)
        print('Validation MCC')
        print(matthews_corrcoef(y_val, y_pred_v))

        # confusion matrix generation external validation
        cm_ex = confusion_matrix(y_val, y_pred_ex)
        print('External Validation MCC')
        print(matthews_corrcoef(y_val, y_pred_ex))


        d = {'col1': ["accuracy training", "f1 training", "efron training", \
                      "cf matrix", "accuracy training", "f1 training", "efron training", \
                      "cf matrix", "accuracy training", "f1 training", "efron training",\
                      "cf matrix", "File Name"],\
             'col2': ["training", "training", "training", "training", "validation", "validation", \
                      "validation", "validation", "external",  "external",  \
                      "external",  "external", " = "], \
             "col3": [acc_t, f1_t, ef_t, cm_t, acc_v, f1_v, ef_v, cm_v, \
                      acc_ex, f1_ex, ef_ex, cm_ex, file_name]
             }
        results = {'col1': failures, \
                   'col2': failure_status
        }

        printout = pd.DataFrame(data=d)
        results_ad_on = pd.DataFrame(data=results)

        with pd.ExcelWriter("C:/Users/trevi/Downloads/BernoulliNB/BernoulliNB/output_18_04_2025.xlsx",
                         mode="a",
                         engine="openpyxl",
                         if_sheet_exists="overlay",
        ) as writer:
            start_row = write_row
            printout.to_excel(writer, sheet_name="LogReg ds_add", startcol=write_line, startrow=start_row)          
            results_ad_on.to_excel(writer, sheet_name="LogReg ds_add", startcol=write_line, startrow=start_row + 18)
        return write_line + 6
    else:
        print(f"file {file_name} was tossed into the flame")
        return write_line

# initialize starting row and column for excel writer
r=40
c=5

# BernoulliNB use this [226, 145, 140, 46, 231, 102, 110, 244, 135, 138, 34]
#random use this [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# LogisticRegression use this [49, 251, 46, 231, 234, 226, 195, 85, 221, 110, 191]
#---------------------------------------RUN SINGLE INDIVIDUAL TRIAL-------------------------------------
c = log_regression([49, 251, 46, 231, 234, 226, 195, 85, 221, 110, 191], [],
                  r'C:\Users\trevi\Downloads\BernoulliNB\BernoulliNB\dataset.xlsx',
                  r'C:/Users/trevi/Downloads/BernoulliNB/BernoulliNB/', 'Log-rerun', c, r)
