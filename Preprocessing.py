import pandas as pd
import numpy as np
from random import randint
from sklearn.model_selection import train_test_split


def load_data(file_name):
    data = pd.read_csv(file_name, sep=',')

    headers = data.columns.values
    data_array = data.to_numpy()

    return headers,data_array


def data_extraction(data,header):
    """
    This function helps with extracting specific data from a dataset.
    This function is specified to exercise 1b from the assignment.
    """

    data[data == 'Pop'] = 1
    data[data == 'Classical'] = 0
        #relabel

    pop_index = data[:, 0] == 1
    classical_index = data[:, 0] == 0
        #gather indexes
    

    pop_data = data[pop_index]
    classical_data = data[classical_index]
        #append

    selected_columns = [0,11,12]
        #this is the desired data in the exercise. 

    pop_data = pop_data[:, selected_columns]
    classical_data = classical_data[:, selected_columns]
    header_label = [header[i] for i in selected_columns]
        #since this is a list and not an array, I had to use list comprehension. 

    return pop_data, classical_data,header_label


def preocessing_data(data_set_1,data_set_2):
    """
    This function is the final step of making the data ready for the ML method.
    It will combine the datasets and seperate them into x and y labels
    """
    
    dataset = np.concatenate((data_set_1, data_set_2), axis=0)

    Y = dataset[:,0]
    X = dataset[:,[1,2]]
        #these are the rows we will be working with. 

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=randint(1,44))

    return X_train, X_test, y_train, y_test