import numpy as np

def linear_regression(datapoints):
    print('\nLinear Regression Part:')
    datapoints = np.array(datapoints)

    X = np.array(datapoints[:, :-1])
    y = np.array(datapoints[:, -1])
    print('\nX is:')
    print(X)
    print('\ny is:')
    print(y)

    X_inverse = np.linalg.pinv(X)
    print('\nX_inverse is:')
    print(X_inverse)

    weights = np.dot(X_inverse, y)
    print('\nweights are:')
    print(weights)

    return weights

def question3():
    answers = {'a': (3 ** .5 + 4) ** .5, 'b': (3 ** .5 - 1) ** .5, 'c': (3 + 4 * 6 ** .5) ** .5, 'd': (9 - 6 ** .5) ** .5}

    for key, rho in answers.items():
        print('\n#############################################################################')
        print(f'key is: {key}, rho is: {rho}')
        print('#############################################################################')
        error_constant = []
        error_linear = []

        for i in range(3):
            datapoints = [[1, -1, 0], [1, rho, 1], [1, 1, 0]]
            del datapoints[i]

            print(f'\n>>>>>>>>>>>epoch: {i}')

            print('\nDatapoints after LOOCV deletin are:')
            print(datapoints)

            weights = linear_regression(datapoints)

            print('\nLOOCV and calculating error part:')

            # Squared error
            for datapoint in datapoints:
                error_linear.append((np.dot(weights, datapoint[:2]) - datapoint[2]) ** 2)
                print(f'\nFor datapoint {datapoint}, the error for h1 is: {error_linear[-1]}')

                error_constant.append((weights[0] - datapoint[2]) ** 2)
                print(f'\nFor datapoint {datapoint}, the error for h0 is: {error_constant[-1]}')

        print('\n#############################################################################')
        print(f'Final answer for key {key}')
        print(f'\nanswer: {key}, error_constant == {np.mean(error_constant)}, error_linear == {np.mean(error_linear)}')
        print('#############################################################################')

question3()