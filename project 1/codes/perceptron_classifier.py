import numpy as np


class Perceptron:
    def __init__(self, learning_rate=1, num_iters=None, w0=None, w1=None, w2=None):
        self.learning_rate = learning_rate
        self.updated = False

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

        # Unlike Pocket, in Perceptron number of iterations isn't determined.
        if self.mode == 'iters-not-specified':

            # The algorithm begins:
            while True:

                # This variable shows whether in an epoch, the weights of the Peceptron model was updated or not.
                self.updated = False

                # In each iteration, we loop through the X by enumerate method.
                for index, x in enumerate(X):

                    # Our raw prediction and then through the activation function:
                    h_raw = np.dot(x, self.w)
                    y_prediction = self._sign_function(h_raw)

                    # Now, the update rule comes up:
                    if not y_prediction == y[index]:

                        # Since the weights are being updates, so this variable becomes True.
                        self.updated = True

                        # Updating:
                        delta_w = self.learning_rate * y[index] * x
                        self.w += delta_w

                # Number of iterations:
                self.num_iters += 1

                # The termination rule. If the weights weren't updated in an epoch, it means that our model has got every datapoint correctly classified, and it didn't need to
                # update the rules. So, therefore, based upon the algorithm, we terminate.
                if not self.updated:
                    break
        elif self.mode == 'iters-specified':
            pass

    def get_final_weights(self):
        return self.w

    def get_num_iterations(self):
        return self.num_iters


def perceptron_classifier(X, y):
    # Experiment #1, <w0, w1, w2> = <0.2, 5.1, 2.3>:
    print('\nPerceptron Classifier - Experiment #1 with weights: <w0, w1, w2> = <0.2, 5.1, 2.3>.\n')

    model = Perceptron(learning_rate=1, w0=0.2, w1=5.1, w2=2.3)

    model.fit(X, y)

    w = model.get_final_weights()

    print(f'Perceptron Classifier - Experiment #1 - The final weights for model #1 are: {w}...')
    print(f'\nPerceptron Classifier - Experiment #1 - The shape of model #1 final weights is: {w.shape}')

    num_iters = model.get_num_iterations()

    print(f'\nPerceptron Classifier - Experiment #1 - Number of iterations that took model #1 to finish learning every datapoint is {num_iters}...')

    # Experiment #2, <w0, w1, w2> = <1.0, 1.0, 1.0>:
    print('\nPerceptron Classifier - Experiment #2 with weights: <w0, w1, w2> = <1.0, 1.0, 1.0>.\n')

    model = Perceptron(learning_rate=1, w0=1.0, w1=1.0, w2=1.0)

    model.fit(X, y)

    w = model.get_final_weights()

    print(f'Perceptron Classifier - Experiment #2 - The final weights for model #2 are: {w}...')
    print(f'\nPerceptron Classifier - Experiment #2 - The shape of model #2 final weights is: {w.shape}')

    num_iters = model.get_num_iterations()

    print(f'\nPerceptron Classifier - Experiment #2 - Number of iterations that took model #2 to finish learning every datapoint is {num_iters}...')

def main():
    pass

if __name__ == '__main__':
    main()