import numpy as np
import pandas as pd


class LogisticRegression:
    def __init__(self, learning_rate, epochs, num_features):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.w = np.zeros(num_features)

    def _sigmoid_function(self, x):
        return 1 / (1 + np.exp(-x))

    def _gradients(self, x, y):
        return (-y * x) / (1 + np.exp(y * np.dot(self.w, x)))

    def _loss_function(self, x, y):
        error = np.log(1 + np.exp(-y * np.dot(x, self.w)))
        error = np.sum(error) / x.shape[0]
        return error

    def train(self, x_train, y_train, x_test, y_test, x_val=None, y_val=None):
        for epoch in range(self.epochs):
            dw = 0.0

            # Calculating gradients of W from all datapoints:
            for i in range(x_train.shape[0]):
                dw += self._gradients(x_train[i], y_train[i])

            # Taking a mean of dw so that it would be normalized:
            dw /= x_train.shape[0]

            # Updating the weights:
            self.w = self.w - (dw * self.learning_rate)

            # Calculating error on training set by the weights of that epoch:
            error_train = self._loss_function(x_train, y_train)

            # If user has given us validation set:
            if x_val is not None:
                error_validation = self._loss_function(x_val, y_val)

                print(f'Epoch: {epoch}, Training Error: {error_train}, Validation Error: {error_validation}')
            else:
                print(f'Epoch: {epoch}, Training Error: {error_train}')

        print('\nThe training has finished, now onto printing results...\n')

        # Calculating error for training set by the final set of weights:
        error_train = self._loss_function(x_train, y_train)

        # Predicting the label of datapoints in training set by the final set of weights:
        y_pred = self.predict(x_train)

        # Finding accuracy score of training set by the final set of weights:
        accuracy_train = self.accuracy_score(y_train, y_pred)

        print(f'Training Error: {error_train}, Training Accuracy: {accuracy_train}')

        if x_val is not None:
            error_validation = self._loss_function(x_val, y_val)
        
            y_pred = self.predict(x_val)

            accuracy_validation = self.accuracy_score(y_val, y_pred)

            print(f'Validation Error: {error_validation}, Validation Accuracy: {accuracy_validation}')

        error_test = self._loss_function(x_test, y_test)

        y_pred = self.predict(x_test)

        accuracy_test = self.accuracy_score(y_test, y_pred)

        print(f'Test Error: {error_test}, Test Accuracy: {accuracy_test}')

    def predict(self, x):
        # Dot product and going through activation function:
        h_raw = np.dot(x, self.w)
        h_raw = self._sigmoid_function(h_raw)

        # Applying threshold:
        y_pred = np.where(h_raw > 0.5, 1, -1)

        return y_pred

    def accuracy_score(self, y, y_pred):
        return (y == y_pred).sum() / y.shape[0]
            
def logistic_regression_titanic(x_train, y_train, x_test, y_test, x_val=None, y_val=None):
    print('\nTuning phase for Titanic dataset, we\'re gonna do some classification and experimenting with validation data.')
    
    # First, we try to do hyper-paramter tuning... We won't show actual hyper-paramter tuning in CMD, since it will take
    # lots of space and forever to finish reading. Thus, we will do some experimenting without showing it.
    # The values we had in mind:
    # learning_rate = [1, 0.1, 0.01, 0.001, 0.0001]
    # epochs = [500, 1000, 2000, 5000, 10000]
    model = LogisticRegression(learning_rate=0.01, epochs=500, num_features=x_train.shape[1])

    print('\nTraining the model for Titanic dataset with validation, for hyper-paramter tuning reasons:\n')

    # Training the model to find the best weight.
    model.train(x_train, y_train, x_test, y_test, x_val, y_val)

    # After finding the best hyper-paramters, we learn and train the model anew:

    print('\nHyper-parameter tuning is finished. We chose the best values for the hyper-paramters, learning rate and number of epochs. Now, onto training the actual model on Titanic dataset:')

    # First, we need to concatenate/join two training and validation arrays and create training dataset anew:
    print('\nConcatenate/Join two training and validation arrays to train the model anew.')

    x_train = np.vstack([x_train, x_val])
    y_train = np.vstack([y_train[:, None], y_val[:, None]]).flatten()

    print('\nx_train is:')
    print(x_train)

    print('\ny_train is:')
    print(y_train)

    print(f'\nThe shape of x_train is: {x_train.shape}')
    print(f'\nThe shape of y_train is: {y_train.shape}')
    
    model = LogisticRegression(learning_rate=0.01, epochs=500, num_features=x_train.shape[1])

    print('\nNow, training the model:\n')

    # Training the model:
    model.train(x_train, y_train, x_test, y_test)

def logistic_regression_gradstudies(x_train, y_train, x_test, y_test):
    print('\nTraining the model on GradStudeis dataset with the best parameter values found in hyper-paramter tuning in Titanic dataset.\n')
    
    # First, we try to do hyper-paramter tuning...
    model = LogisticRegression(learning_rate=0.01, epochs=500, num_features=x_train.shape[1])

    # Training the model:
    model.train(x_train, y_train, x_test, y_test)
    
def main():
    pass

if __name__ == '__main__':
    pass