import numpy as np


class Pocket:
    def __init__(self, learning_rate=1, num_iters=2000, w0=None, w1=None, w2=None):
        self.learning_rate = learning_rate
        self.best_w = None
        self.best_loss = np.inf

        if w0 == None and w1 == None and w2 == None:
            self.w = self._initialize_weights()
        else:
            self.w = np.array([w0, w1, w2])

        if num_iters == None:
            self.mode = 'iters-not-specified'
            self.num_iters = 0
        else:
            self.mode = 'iters-specified'
            self.num_iters = num_iters

    def _initialize_weights(self):
        return np.array([np.random.uniform(-2, 2), np.random.uniform(-2, 2), np.random.uniform(-2, 2)])

    def _sign_function(self, x):
        return np.where(x >= 0, 1, -1)

    def fit(self, X, y):

        # Unlike Perceptron, in Pocket number of iterations is determined.
        if self.mode == 'iters-specified':

            # We first have to get loss for initial weights and assign to the best_loss and the initial weights to the best_w.
            # We've determined the initial best_loss as +inifinite, so any loss value will replace it.
            loss = self.loss_function(X, y)

            if loss < self.best_loss:
                self.best_loss = loss
                self.best_w = self.w

            # Now, the algorithm begins its work.
            for _ in range(self.num_iters):

                # In each iteration, we loop through the X by enumerate method.
                for index, x in enumerate(X):

                    # Our raw prediction and then through the activation function:
                    h_raw = np.dot(x, self.w)
                    y_prediction = self._sign_function(h_raw)

                    # Now, the update rule comes up:
                    if not y_prediction == y[index]:

                        # Updating:
                        delta_w = self.learning_rate * y[index] * x
                        self.w += delta_w

                        # Get loss for the updated weights:
                        loss = self.loss_function(X, y)

                        # Compare it with the best loss. If the current loss is lower, replace the best loss with the current loss and also, the weights.
                        if loss < self.best_loss:
                            self.best_loss = loss
                            self.best_w = self.w
        elif self.mode == 'iters-not-specified':
            pass

    def predict(self, X, w=None):

        # Predictions of the given weights on the whole dataset.
        y_predictions = []

        for index, x in enumerate(X):

            # If the user gives this method a list/array of weights, we will predict with that. If not, we will predict using self.w, the current weight in training phasse.
            if w is not None:
                h_raw = np.dot(x, w)
            else:
                h_raw = np.dot(x, self.w)
            
            y_prediction = self._sign_function(h_raw)
            y_predictions.append(y_prediction)

        # Converting the predictions from list to numpy array for consistency matters.
        y_predictions = np.array(y_predictions)

        # We have to reshape so that the comparison between them is possible.
        y_predictions = np.reshape(y_predictions, (-1, 1))

        return y_predictions

    def loss_function(self, X, y, w=None):

        # We first make predictions for X. If no value for w is assigned, then, we will predict using current weight.
        y_predictions = self.predict(X, w)

        # Compute loss for the weights given.
        loss = (y != y_predictions).sum() / y.shape[0]

        return loss

    def get_best_weights(self):
        return self.best_w

    def accuracy_score(self, X, y, w=None):

        # We first make predictions for X. If no value for w is assigned, then, we will predict using current weight.
        y_predictions = self.predict(X, w)

        # Instead of loss, we compute accuracy score. We could also do like this: accuracy = 1 - loss.
        score = (y == y_predictions).sum() / y.shape[0]

        return score


def pocket_classifier(X, y):
    # Before getting into classification with Pocket, we have to add the datapoints:
    print('\nAdding two extra datapoints.\n')

    X = np.vstack([X, [1, 5, 1]])
    y = np.vstack([y, [-1]])

    X = np.vstack([X, [1, 3, 1.5]])
    y = np.vstack([y, [-1]])

    print('X after addition is:')
    print(X)

    print('\ny after addition is:')
    print(y)

    print(f'\nThe shape of X after addition is: {X.shape}')
    print(f'\nThe shape of y after addition is: {y.shape}')

    # Experiment #1, <w0, w1, w2> = <0.2, 5.1, 2.3>:
    print('\nPocket Classifier - Experiment #1 with weights: <w0, w1, w2> = <0.2, 5.1, 2.3>.\n')

    model = Pocket(learning_rate=1, num_iters=2000, w0=0.2, w1=5.1, w2=2.3)

    model.fit(X, y)

    best_w = model.get_best_weights()

    print(f'Pocket Classifier - Experiment #1 - The best weights for model #1 are: {best_w}...')
    print(f'\nPocket Classifier - Experiment #1 - The shape of model #1 best weights is: {best_w.shape}')

    best_score = model.accuracy_score(X, y, best_w)

    print(f'\nPocket Classifier - Experiment #1 - Accuracy score of model #1 with best weights is: {best_score}...')

    best_loss = model.loss_function(X, y, best_w)

    print(f'\nPocket Classifier - Experiment #1 - Loss value of model #1 with best weights is: {best_loss}...')

    # Experiment #2, <w0, w1, w2> = <1.0, 1.0, 1.0>:
    print('\nPocket Classifier - Experiment #2 with weights: <w0, w1, w2> = <1.0, 1.0, 1.0>.\n')

    model = Pocket(learning_rate=1, w0=1.0, w1=1.0, w2=1.0)

    model.fit(X, y)

    best_w = model.get_best_weights()

    print(f'Pocket Classifier - Experiment #2 - The best weights for model #2 are: {best_w}...')
    print(f'\nPocket Classifier - Experiment #2 - The shape of model #2 best weights is: {best_w.shape}')

    best_score = model.accuracy_score(X, y, best_w)

    print(f'\nPocket Classifier - Accuracy score of model #2 with best weights is: {best_score}...')

    best_loss = model.loss_function(X, y, best_w)

    print(f'\nPocket Classifier - Experiment #2 - Loss value of model #1 with best weights is: {best_loss}...')

def main():
    pass

if __name__ == '__main__':
    main()