"""
Basic functions for idiot
"""

import numpy as np
import pandas as pd
import pickle
from os import path

from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler

import scikitplot as skplt
import seaborn as sns
import matplotlib.pyplot as plt
import os



def display_results(y_test, y_pred, run_time):
    """
    Display performance metrics of machine learning models
    """
    for i in range(len(y_pred)):
        y_pred[i] = int(round(y_pred[i]))  # just in case linear regression is used

    # print("Mean Absolute Error - ", metrics.mean_absolute_error(y_test, y_pred))
    # print("Mean Squared Error - ", metrics.mean_squared_error(y_test, y_pred))
    # print("Root Mean Squared Error - ", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    # print("R2 Score - ", metrics.explained_variance_score(y_test, y_pred)*100)
    # print("Accuracy - ", accuracy_score(y_test, y_pred)*100)
    MAE_loss = metrics.mean_absolute_error(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average="weighted")
    recall = metrics.recall_score(y_test, y_pred, average="weighted")
    f1_score = 2 * (precision * recall) / (precision + recall)

    print("\nMean Absolute Error - ", MAE_loss)
    print("Precision - ", precision * 100)
    print("Recall - ", recall * 100)
    print("F1 score - ", f1_score * 100)
    print("Accuracy - ", accuracy_score(y_test, y_pred) * 100)
    print("Running training - ", run_time)

def save_model(file_name, model):
    """
    Save machine learning model to file_name
    """
    if not path.isfile(file_name):
        # saving the trained model to disk
        with open(file_name, 'wb') as file:
            pickle.dump(model, file)
        print("Saved model to disk")
    else:
        print("Model already saved")
        
        
def reading_data_full(data_path_full=None, data_path_feature=None, n_start=None, n_end=None):
    # Reading datasets
    dfs = []
    for i in range(n_start,n_end):
        path = data_path_full  # There are 4 input csv files
        dfs.append(pd.read_csv(path.format(i), header = None))
    all_data = pd.concat(dfs).reset_index(drop=True)  # Concat all to a single df

    # This csv file contains names of all the features
    df_col = pd.read_csv(data_path_feature, encoding='ISO-8859-1')
    
    # Making column names lower case, removing spaces
    df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower())

    # Renaming our dataframe with proper column names
    all_data.columns = df_col['Name']
    
    
    return all_data

def preprocessing_data_unsw_full(all_data=None, binary_classify=None, normalized = None):
    # Filling null
    # We don't have "normal" values for "attack_cat", so we must fill Null values with "normal"
    all_data['attack_cat'] = all_data.attack_cat.fillna(value='Normal')
   
    all_data['ct_flw_http_mthd'] = (all_data.ct_flw_http_mthd.fillna(value=0)).astype(int)
    
    # Even though it's a binary column, but there're values like 2 and 4
    all_data['is_ftp_login'] = (all_data.is_ftp_login.fillna(value=0)).astype(int)
    
    # Removing empty space and converting it to numerical
    all_data['ct_ftp_cmd'] = all_data['ct_ftp_cmd'].replace(to_replace=' ', value=0).astype(int)
    
    # The col "is_ftp_login" has few wrong values like 2, 4. It should only have 0 and 1, If the ftp session is accessed by user and password then 1 else 0. Need to fix this.
    all_data['is_ftp_login'] = np.where(all_data['is_ftp_login']>1, 1, all_data['is_ftp_login'])
    
    # removing all the "-" and replacing those with "None"
    all_data['service'] = all_data['service'].apply(lambda x:"None" if x=="-" else x)

    all_data['attack_cat'] = all_data['attack_cat'].replace('Backdoors','Backdoor', regex=True)
    
    
    # creating new features
    all_data['network_bytes'] = all_data['sbytes'] + all_data['dbytes']
        
    all_data = all_data.drop(columns=['srcip', 'sport', 'dstip', 'dsport', 'sbytes', 'dbytes'])
    # select all columns of numeric types (including the label column if applicable)
    num_col = all_data.select_dtypes(include='number').columns
    
    # select all columns of NON-numeric types (including the label column if applicable)
    temp_col = all_data.columns.difference(num_col)
    temp_col = temp_col.drop(['attack_cat'])  # remove the lable column

    # creating a dataframe with only categorical attributes
    data_cat = all_data[temp_col].copy()

    # ONE HOT ENCODING: Convert data_cat into a df with numeric values by using pd.get_dummies() function
    data_cat = pd.get_dummies(data_cat, columns=temp_col)

    # first, concatenate data and data_cat
    all_data = pd.concat([all_data, data_cat], axis=1)

    # then drop the non-numeric columns
    all_data.drop(columns=temp_col, inplace=True)
    
    if normalized:
        all_data = normalization(all_data.copy(), num_col)
    
    if not binary_classify:
        all_data = numbering_label_unsw(all_data)
    all_data = all_data.drop(columns=['attack_cat'])
    
    return all_data


def preprocessing_data_unsw(data_path=None, normalized=False, binary_classify=False):
    """
    This work for multiple class (UNSW dataset)
    """
    data = pd.read_csv(data_path)
    data = data.drop(columns=['id'])
    data['service'].replace('-', 'other', inplace=True)

    # select all columns of numeric types (including the label column if applicable)
    num_col = data.select_dtypes(include='number').columns

    # select all columns of NON-numeric types (including the label column if applicable)
    temp_col = data.columns.difference(num_col)
    temp_col = temp_col.drop(['attack_cat'])  # remove the lable column

    # creating a dataframe with only categorical attributes
    data_cat = data[temp_col].copy()

    # ONE HOT ENCODING: Convert data_cat into a df with numeric values by using pd.get_dummies() function
    data_cat = pd.get_dummies(data_cat, columns=temp_col)

    # first, concatenate data and data_cat
    data = pd.concat([data, data_cat], axis=1)

    # then drop the non-numeric columns
    data.drop(columns=temp_col, inplace=True)

    if normalized:
        data = normalization(data.copy(), num_col)
    
    if not binary_classify:
        data = numbering_label_unsw(data)
    data = data.drop(columns=['attack_cat'])

    return data


def normalization(df, col):
    """
    Using minmax sacler for normalizing data
    """
    minmax_scale = MinMaxScaler(feature_range=(0, 1))

    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = minmax_scale.fit_transform(arr.reshape(len(arr), 1))
    return df




def numbering_label_unsw(data):
    label_number = np.zeros(len(data['attack_cat']))
    label_text = data['attack_cat']
    # print(label_text)
    for i in range(len(label_number)):
        # print('label_text[i]: ', label_text[i])
        if label_text[label_text.index[i]] == 'DoS':
            label_number[i] = 1
        elif label_text[label_text.index[i]] == 'Exploits':
            label_number[i] = 2
        elif label_text[label_text.index[i]] == 'Fuzzers':
            label_number[i] = 3
        elif label_text[label_text.index[i]] == 'Generic':
            label_number[i] = 4
        elif label_text[label_text.index[i]] == 'Reconnaissance':
            label_number[i] = 5
        elif label_text[label_text.index[i]] == 'Analysis':
            label_number[i] = 6
        elif label_text[label_text.index[i]] == 'Backdoor':
            label_number[i] = 7
        elif label_text[label_text.index[i]] == 'Worms':
            label_number[i] = 8
        elif label_text[label_text.index[i]] == 'Shellcode':
            label_number[i] = 9
        else:
            label_number[i] = 0

    data['label'] = label_number.astype(int)
    return data


def decode_label(y=None, binary_classify=None):
    y = y.astype(str)
    if binary_classify:
        for i in range(len(y.values)):
            if y.values[i] == '0' or y.values[i] == '0.0':
                y.values[i] = 'Normal'
            else:
                y.values[i] = 'Abnormal'
    else:
        y = y.astype(str)
        for i in range(len(y.values)):
            if y.values[i] == '1':
                y.values[i] = 'DoS'
            elif y.values[i] == '2':
                y.values[i] = 'Exploits'
            elif y.values[i] == '3':
                y.values[i] = 'Fuzzers'
            elif y.values[i] == '4':
                y.values[i] = 'Generic'
            elif y.values[i] == '5':
                y.values[i] = 'Reconnaissance'
            elif y.values[i] == '6':
                y.values[i] = 'Analysis'
            elif y.values[i] == '7':
                y.values[i] = 'Backdoor'
            elif y.values[i] == '8':
                y.values[i] = 'Worms'
            elif y.values[i] == '9':
                y.values[i] = 'Shellcode'
            else:
                y.values[i] = 'Normal'
    return y



def align_test_dataset(data_test, data_train):
    drop_cols = [x for x in data_test if x not in data_train]  # Drop cols in data_test without in data_train
    add_cols = [x for x in data_train if x not in data_test]  # Add cols in data_test with in data_train

    # Function return location cols in data_train
    def get_loc_col(data=None, features=None):
        for i in range(len(data)):
            if data.columns[i] in features:
                return i

    data_test = data_test.drop(columns=drop_cols)

    i = 0
    while i < len(add_cols):
        data_test.insert(loc=get_loc_col(data_train, add_cols[i]), column=add_cols[i], value=0)
        i = i + 1
    return data_test

"""Utils function for plot"""

def confusion_matrix(y_test, y_pred, file_name, binary_classify, types):
    import os
    if not os.path.exists("plots/Decisiontree/binary/"):
        os.makedirs("plots/Decisiontree/binary/")
    if not os.path.exists("plots/Decisiontree/multiple/"):
        os.makedirs("plots/Decisiontree/multiple/")
    if not os.path.exists("plots/Randomforest/binary/"):
        os.makedirs("plots/Randomforest/binary/")
    if not os.path.exists("plots/Randomforest/multiple/"):
        os.makedirs("plots/Randomforest/multiple/")
    if binary_classify:
        skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                            normalize=False,
                                            title=" ",
                                            cmap="Blues",
                                            text_fontsize="large",
                                            figsize=(10.2, 7))
        if file_name[:22] == "RandomForestClassifier":
            plt.savefig("plots/Randomforest/binary/"   + file_name + types  +".pdf")
        else:
            plt.savefig("plots/Decisiontree/binary/"   + file_name + types +".pdf")
    else:
        skplt.metrics.plot_confusion_matrix(y_test, y_pred,
                                            normalize=False,
                                            x_tick_rotation=10,
                                            title=" ",
                                            cmap="Purples",
                                            text_fontsize="large",
                                            figsize=(10.2, 7))
        if not os.path.exists("plots/multi/"):
            os.makedirs("plots/multi/")
        plt.savefig("plots/multi/" +types +".pdf")
            
def visualize_data(df):
    # Plotting target label distribution
    plt.figure()
    plt.title("Binary-class distribution of dataset")
    df['label'].value_counts().plot(kind="bar", color='b', label="dataset")
    #test['label'].value_counts().plot(kind="bar", color='orange', label="test")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("../plots/visualize/Binary-class distribution of dataset.pdf")
    
    # Plotting target label distribution
    plt.figure()
    plt.title("Multi-class distribution of dataset")
    df['attack_cat'].value_counts().plot(kind="bar", color='b', label="dataset")
    #test['label'].value_counts().plot(kind="bar", color='orange', label="test")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("../plots/visualize/multi-class distribution of dataset")
    
