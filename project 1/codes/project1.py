import numpy as np
import pandas as pd

from dataset_preprocess import dataset_preprocess
from perceptron_classifier import perceptron_classifier
from pocket_classifier import pocket_classifier


def main():
    # Data and Preprocessing stuff:
    print('\nDataset and preprocessing phase:')
    X, y = dataset_preprocess()

    # Now that our data is ready, time to get classifying.

    # First, with Perceptron:
    print('\nPerceptron Classifier:')
    perceptron_classifier(X, y)

    # And now, with Pocket:
    print('\nPocket Classifier:')
    pocket_classifier(X, y)

if __name__ == '__main__':
    main()