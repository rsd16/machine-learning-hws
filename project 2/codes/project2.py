from dataset_preprocess import dataset_preprocess_titanic, dataset_preprocess_gradstudies
from logistic_regression import logistic_regression_titanic, logistic_regression_gradstudies


def main():
    # Data and Preprocessing stuff:
    print('\nDataset and preprocessing phase:')

    # First, Titanic dataset:
    print('\nTittanic dataset:')
    
    x_train, y_train, x_test, y_test, x_val, y_val = dataset_preprocess_titanic()

    # Now that our data is ready, time to get classifying with Logistic Regression:
    logistic_regression_titanic(x_train, y_train, x_test, y_test, x_val, y_val)

    # Data and Preprocessing stuff:
    print('\nDataset and preprocessing phase:')

    # Second, GradStudies dataset:
    print('\nGradStudies dataset:')
    
    x_train, y_train, x_test, y_test = dataset_preprocess_gradstudies()

    # Now that our data is ready, time to get classifying with Logistic Regression:
    logistic_regression_gradstudies(x_train, y_train, x_test, y_test)

if __name__ == '__main__':
    main()