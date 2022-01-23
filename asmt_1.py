import numpy as np
from sklearn.model_selection import KFold
import pandas as pd
import time

# Global variables
data_total = [0]
d_feature = [0]
d_class = [0]
count_df = [0]


# This function should open a data file in csv, and transform it into a usable format
def load_data(filepath):
    global data_total, d_feature, d_class
    data_total = pd.read_csv(filepath, sep=',')
    d_feature = data_total.iloc[:, 0:29]
    d_class = data_total['Grade']
    return


# This function should split a data set into a training set and hold-out test set
# res[0] : training data [0-518] 80% of the total data
# res[1] : test data [519-648] 20% of the total data
def split_data():
    arr_train = list(range(0, 519))
    arr_test = list(range(519, 649))
    res = [0]
    res[0] = arr_train
    res.append(arr_test)

    return res


# %%

# This function should build a supervised NB model
def train(train_index):
    global d_feature
    global d_class
    global data_total

    train_inst = data_total.iloc[train_index]

    # region [make count table]
    global count_df

    count_df[0] = train_inst.groupby(['Grade', 'school']).size().reset_index(name='count')

    # test
    # print("[count_school]"); print(count_df[0])
    # access to conditional value
    # res = count_df[0].loc[(count_df[0]['Grade'] == 'A') & (count_df[0]['school'] == 'GP')].at[0,'count']
    # print("res", res)
    ##
    count_df.append(train_inst.groupby(['Grade', 'sex']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'address']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'famsize']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'Pstatus']).size().reset_index(name='count'))

    count_df.append(train_inst.groupby(['Grade', 'Medu']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'Fedu']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'Mjob']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'Fjob']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'reason']).size().reset_index(name='count'))

    count_df.append(train_inst.groupby(['Grade', 'guardian']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'traveltime']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'studytime']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'failures']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'schoolsup']).size().reset_index(name='count'))

    count_df.append(train_inst.groupby(['Grade', 'famsup']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'paid']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'activities']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'nursery']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'higher']).size().reset_index(name='count'))

    count_df.append(train_inst.groupby(['Grade', 'internet']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'romantic']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'famrel']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'freetime']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'goout']).size().reset_index(name='count'))

    count_df.append(train_inst.groupby(['Grade', 'Dalc']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'Walc']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'health']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby(['Grade', 'absences']).size().reset_index(name='count'))
    count_df.append(train_inst.groupby('Grade').size().reset_index(name='count'))  # 29

    # print(count_df)
    # test

    # print("모든 count table")
    # print(count_df)
    # print("[count_Grade]")
    # print(count_df[29])
    # print("[Grade at : ]", count_df[29].iat[0, 1])

    # endregion

    return


# %%

# This function should predict the class for an instance or a set of instances, based on a trained model
def predict(test_index):
    GradeArr = ["A+", "A", "B", "C", "D", "F"]
    AttbArr = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
               'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
               'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

    test_inst = data_total.iloc[test_index]
    grade_tot = count_df[29].sum(numeric_only=True).iat[0]
    predicted_class = [0]
    i = 0

    for inst in test_inst.iterrows():

        max_y = float('-inf')  # minimum int
        est_res = ""

        for g in GradeArr:

            grade_cnt = count_df[29].loc[count_df[29]['Grade'] == g].iat[0, 1]
            est_y = np.log(grade_cnt / grade_tot)

            for att_idx in range(29):
                df_condition = count_df[att_idx].loc[
                    (count_df[att_idx]['Grade'] == g) & (count_df[att_idx][AttbArr[att_idx]] == inst[1].array[att_idx])]

                if df_condition.empty == True:
                    cnt_test = 0  # 0 - smoothing?
                else:
                    cnt_test = df_condition.iat[0, 2]

                if grade_cnt == 0 or grade_cnt == 0.0:
                    est_y += 0  # smoothing?
                else:
                    est_y += np.log(cnt_test / grade_cnt)

            if est_y >= max_y:
                max_y = est_y
                est_res = g

        if i == 0:
            predicted_class[0] = est_res
        else:
            predicted_class.append(est_res)
        i += 1

    return predicted_class


# %%

def predict_fair(test_index):
    GradeArr = ["A+", "A", "B", "C", "D", "F"]
    AttbArr = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian',
               'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
               'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

    test_inst = data_total.iloc[test_index]
    grade_tot = count_df[29].sum(numeric_only=True).iat[0]
    predicted_class = [0]
    i = 0

    for inst in test_inst.iterrows():

        max_y = float('-inf')  # minimum int
        est_res = ""

        for g in GradeArr:

            grade_cnt = count_df[29].loc[count_df[29]['Grade'] == g].iat[0, 1]
            est_y = np.log(grade_cnt / grade_tot)


            for att_idx in range(29-7):

                df_condition = count_df[att_idx].loc[
                    (count_df[att_idx]['Grade'] == g) & (count_df[att_idx][AttbArr[att_idx]] == inst[1].array[att_idx])]

                if df_condition.empty == True:
                    cnt_test = 0  # 0 - smoothing?
                else:
                    cnt_test = df_condition.iat[0, 2]

                # remove 7 features 1 4 5 6 7 8 10
                if att_idx == 1 or att_idx == 4 or att_idx == 5 or att_idx == 6 or att_idx == 7 or att_idx == 8 or att_idx == 10 or grade_cnt == 0 or grade_cnt == 0.0:
                    est_y += 0
                else:
                    est_y += np.log(cnt_test / grade_cnt)

            if est_y >= max_y:
                max_y = est_y
                est_res = g

        if i == 0:
            predicted_class[0] = est_res
        else:
            predicted_class.append(est_res)
        i += 1

    return predicted_class



# This function should evaluate a set of predictions in terms of accuracy
def evaluate(test_index, prd_res):
    real_res = d_class.iloc[test_index]
    tot_cnt = real_res.size
    i = 0
    correct_cnt = 0

    for real in real_res:
        if real == prd_res[i]:
            correct_cnt += 1
            # print("correct index: ", i)
        # else : print("incorrect index: ", i)
        i += 1

    accr = correct_cnt / tot_cnt

    return accr

# %%


# region [Main]
load_data('./Data/student.csv')

# region [HOLD-OUT // SPLIT 80:20 // Accuracy : 0.34]
res = split_data()
train(res[0])
prd_res = predict(res[1])  # returns predicted array (test instances')
accr_res = evaluate(res[1], prd_res)
print("Holdout Accuracy: ", accr_res)
# endregion


# region [CROSS-VALIDATION // Accuracy : 0.49]
start_time = time.time()

N_SPLIT = 65
kf = KFold(n_splits=N_SPLIT, random_state=0)  # n_fold=65, 649

i = 0
accr_df = [0]

for train_index, test_index in kf.split(data_total):
    train(train_index)
    prd_res = predict(test_index)

    #Fair way
    #prd_res = predict_fair(test_index)

    accr_res = evaluate(test_index, prd_res)
    if i == 0:
        accr_df[i] = accr_res
    else:
        accr_df.append(accr_res)
    print("Iteration : ", i)
    i += 1

print("[Cross-validation Result]")
print("[Running Time]: ", (time.time() - start_time))
print("[N_SPLIT]: ", N_SPLIT)
print("[Accuracy Array]: ", accr_df)
print("[Averaged Accuracy]: ", np.mean(accr_df))

# endregion


# endregion
