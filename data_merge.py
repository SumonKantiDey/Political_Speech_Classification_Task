import pandas as pd
from sklearn import model_selection
import numpy as np

def data_merge():
    dat_speeches1 = pd.read_csv('input/dat_speeches_043114_immi_h_ascii_07212021.csv')
    target1 = pd.read_csv('input/hand_coding_task_house_1000_07162021_lite.csv')
    dat_speeches2 = pd.read_csv('input/dat_speeches_043114_immi_s_ascii_07202021.csv')
    target2 = pd.read_csv('input/hand_coding_task_senate_1000_07032021_lite.csv')

    df1 = dat_speeches1.merge(target1, how="left", on=["doc_id"])
    df2 = dat_speeches2.merge(target2, how="left", on=["doc_id"])

    print("dataset1 length = {} dataset2 length = {}".format(len(df1),len(df2)))

    notnull_df1 = df1[df1['immigration'].notnull()]
    notnull_df1 = notnull_df1.reset_index(drop=True)
    notnull_df2 = df2[df2['immigration'].notnull()]
    notnull_df2 = notnull_df2.reset_index(drop=True)

    print("label df1 = {} label df2 = {}".format(len(notnull_df1),len(notnull_df2)))

    data = pd.concat([
        notnull_df1[['text', 'immigration']],
        notnull_df2[['text', 'immigration']],
        ],axis = 0).reset_index(drop=True)

    data = data.sample(frac=1).reset_index(drop=True)

    print("label data lenght = {}".format(len(data)))

    null_df1 = df1[df1['immigration'].isnull()]
    null_df1 = null_df1.reset_index(drop=True)
    null_df2 = df2[df2['immigration'].isnull()]
    null_df2 = null_df2.reset_index(drop=True)

    print("without label df1 = {} without label df2 = {}".format(len(null_df1),len(null_df2)))
    data.to_csv('input/data.csv',index=False)

    without_labe_data = pd.concat([
        null_df1[['text', 'immigration']],
        null_df2[['text', 'immigration']],
        ],axis = 0).reset_index(drop=True)

    without_labe_data = without_labe_data.sample(frac=1).reset_index(drop=True)

    print("not label data lenght = {}".format(len(without_labe_data)))
    without_labe_data.to_csv('input/unseen_data.csv',index=False)

def data_split():
    data = pd.read_csv('input/data.csv')
    #train, validate, test = np.split(data.sample(frac=1, random_state=42),[int(.8*len(data)), int(.9*len(data))])
    train, test = model_selection.train_test_split(
        data, 
        test_size=0.1, 
        random_state=42, 
        stratify=data.immigration.values
    )

    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print(len(train), len(test))
    print(train.immigration.value_counts())
    print(test.immigration.value_counts())
    train.to_csv('input/train.csv',index=False)
    test.to_csv('input/test.csv',index=False)
    print(train[train.isnull().any(axis=1)])
    print(test[test.isnull().any(axis=1)])


if __name__ == "__main__":
    data_split()