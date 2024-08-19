import numpy as np
import pandas as pd


def dataset_preprocess():
    # Read and print the dataset:
    print('\nReading the Data from *.csv files.\n')

    X = pd.read_csv('Bankruptcy_att.csv', header=None)
    y = pd.read_csv('Bankruptcy_class.csv', header=None)

    print('X is:')
    print(X)

    print('\ny is:')
    print(y)

    print(f'\nThe shape of X is: {X.shape}')
    print(f'\nThe shape of y is: {y.shape}')
    
    # Add the bias feature/column to the beginning of our dataset:
    print('\nAdding Bias to the said Data.\n')

    X.rename(columns={0: 1, 1: 2}, inplace=True)

    X.insert(loc=0, column=0, value=[1] * len(X))

    print('X is:')
    print(X)

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

    return X, y

def main():
    pass

if __name__ == '__main__':
    main()