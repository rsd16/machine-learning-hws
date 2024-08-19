import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def dataset_preprocess_titanic():
    print('\nReading the Data from *.csv file for Titanic Dataset.\n')

    # Read and print the dataset:
    X = pd.read_csv('titanicdata.csv')

    print('X is:')
    print(X)

    print(f'\nThe shape of X is: {X.shape}')

    # Remove the label column from the dataset and store it in a variable:
    y = X['Survived']

    print('\ny is:')
    print(y)

    print('\nThe count distribution of the labels is:')
    print(y.value_counts())

    X.drop('Survived', axis=1, inplace=True)

    print('\nAfter deleting label column, X is:')
    print(X)

    print('\nColumns/Features of X are:')
    print(X.columns)

    # Now for some normalization all over features of X:
    X = X.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    print('\nAfter normalization, X is:')
    print(X)

    # Now changing 0 values to -1 in y (labels):
    y.replace(0, -1, inplace=True)

    print('\nAfter changing 0 to -1 in y (or labels), y is:')
    print(y)

    # Add the bias feature/column to the beginning of our dataset:
    X.insert(loc=0, column='x0', value=[1] * len(X))

    print('\nAfter adding bias to the X, X is:')
    print(X)

    print('\nColumns/Features of X are:')
    print(X.columns)

    print(f'\nThe shape of X is: {X.shape}')

    # For sake of speedy learning, convert our dataframes to numpy arrays:
    print('\nConverting our Data from DataFrame format to Numpy array.\n')

    X = np.array(X)
    y = np.array(y)

    print('X is:')
    print(X)

    print('\ny is:')
    print(y)

    print(f'\nThe shape of X is: {X.shape}')
    print(f'\nThe shape of y is: {y.shape}')

    # Now for splitting the consisting data into train and test:
    print('\nSplitting the data into training and testing data.\n')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)

    print('\nx_train is:')
    print(x_train)

    print('\ny_train is:')
    print(y_train)

    print('\nx_val is:')
    print(x_val)

    print('\ny_val is:')
    print(y_val)

    print('\nx_test is:')
    print(x_test)

    print('\ny_test is:')
    print(y_test)

    print(f'\nThe shape of x_train is: {x_train.shape}')
    print(f'\nThe shape of y_train is: {y_train.shape}')

    print(f'\nThe shape of x_val is: {x_val.shape}')
    print(f'\nThe shape of y_val is: {y_val.shape}')

    print(f'\nThe shape of x_test is: {x_test.shape}')
    print(f'\nThe shape of y_test is: {y_test.shape}')

    values, counts = np.unique(y_train, return_counts=True)

    print('\nValues and Counts of them for y_train is (class distribution):')
    print(np.asarray((values, counts)).T)

    values, counts = np.unique(y_val, return_counts=True)

    print('\nValues and Counts of them for y_val is (class distribution):')
    print(np.asarray((values, counts)).T)

    values, counts = np.unique(y_test, return_counts=True)

    print('\nValues and Counts of them for y_test is (class distribution):')
    print(np.asarray((values, counts)).T)

    return x_train, y_train, x_test, y_test, x_val, y_val

def dataset_preprocess_gradstudies():
    print('\nReading the Data from *.csv file for GradStudies Dataset.\n')

    # Read and print the dataset:
    X = pd.read_csv('gradstudies.csv')

    print('X is:')
    print(X)

    print(f'\nThe shape of X is: {X.shape}')
    print(X.columns)

    # Remove the label column from the dataset and store it in a variable:
    y = X['y']

    print('\ny is:')
    print(y)

    print('\nThe count distribution of the labels is:')
    print(y.value_counts())

    X.drop('y', axis=1, inplace=True)

    print('\nAfter deleting label column, X is:')
    print(X)

    print('\nColumns/Features of X are:')
    print(X.columns)

    # Now for some normalization all over features of X:
    X = X.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

    print('\nAfter normalization, X is:')
    print(X)

    # Now changing 0 values to -1 in y (labels):
    y.replace(0, -1, inplace=True)

    print('\nAfter changing 0 to -1 in y (or labels), y is:')
    print(y)

    # Add the bias feature/column to the beginning of our dataset:
    X.insert(loc=0, column='x0', value=[1] * len(X))

    print('\nAfter adding bias to the X, X is:')
    print(X)

    print('\nColumns/Features of X are:')
    print(X.columns)

    print(f'\nThe shape of X is: {X.shape}')

    # For sake of speedy learning, convert our dataframes to numpy arrays:
    print('\nConverting our Data from DataFrame format to Numpy array.\n')

    X = np.array(X)
    y = np.array(y)

    print('X is:')
    print(X)

    print('\ny is:')
    print(y)

    print(f'\nThe shape of X is: {X.shape}')
    print(f'\nThe shape of y is: {y.shape}')

    # Now for splitting the consisting data into train and test:
    print('\nSplitting the data into training and testing data.\n')

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    print('\nx_train is:')
    print(x_train)

    print('\ny_train is:')
    print(y_train)

    print('\nx_test is:')
    print(x_test)

    print('\ny_test is:')
    print(y_test)

    print(f'\nThe shape of x_train is: {x_train.shape}')
    print(f'\nThe shape of y_train is: {y_train.shape}')

    print(f'\nThe shape of x_test is: {x_test.shape}')
    print(f'\nThe shape of y_test is: {y_test.shape}')

    values, counts = np.unique(y_train, return_counts=True)

    print('\nValues and Counts of them for y_train is (class distribution):')
    print(np.asarray((values, counts)).T)

    values, counts = np.unique(y_test, return_counts=True)

    print('\nValues and Counts of them for y_test is (class distribution):')
    print(np.asarray((values, counts)).T)

    return x_train, y_train, x_test, y_test

def main():
    pass

if __name__ == '__main__':
    main()