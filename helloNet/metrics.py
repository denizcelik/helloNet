import numpy as np


def cross_entropy_binary():
    pass


def cross_entropy_categorical():
    pass


def compute_accuracy_binary(activations, labels):

    # Activations as probabilities
    probs = activations

    # Get boolean equivalent of probabilities
    probs_bools = probs[0] > 0.5

    # Create ones vector to turn booleans vector to predictions
    ones_to_convert = np.ones(probs.shape[1])

    # Calculate predictions using boolean vector
    preds = probs_bools * ones_to_convert

    # Reshape predictions as (1,m) column vector
    preds = preds.reshape(1, preds.shape[0])

    # Get number of set examples
    m_set = labels.shape[1]

    # Calculate number of true predictions
    num_true_preds = np.sum(preds == labels)

    # Calculate accuracy
    accuracy = num_true_preds / m_set

    # Return accuracy
    return accuracy


def compute_accuracy_categorical():
    pass


def compute_fscore():
    pass
