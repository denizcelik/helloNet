import numpy as np


def flatten_data(X_set):

    # Reshape set example as [multip. of other dims, number of examples]
    X_set_flatten = X_set.reshape(X_set.shape[0], -1).T

    # Print new shape
    print(f"Flattened data - new shape: {X_set_flatten.shape}")

    # Return variable
    return X_set_flatten


def normalize_data(X_set):

    # Get the feature-wise minimum to scale
    feature_mins = np.min(X_set, axis=1, keepdims=1)

    # Get the feature-wise maximum to scale
    feature_maxs = np.max(X_set, axis=1, keepdims=1)

    # Normalize image (scale min and max to between 0 and 1)
    X_normalized = (X_set - feature_mins) / (feature_maxs - feature_mins)

    processed_min = np.min(X_normalized)
    processed_max = np.max(X_normalized)

    print(f"Normalized data - min: {processed_min:.2f} - max: {processed_max:.2f}")

    # Return variable
    return X_normalized


def normalize_image_data(X_set):

    # Normalize data example
    X_normalized = X_set / 255

    processed_min = np.min(X_normalized)
    processed_max = np.max(X_normalized)

    print(f"Normalized data - min: {processed_min:.2f} - max: {processed_max:.2f}")

    # Return variable
    return X_normalized


def standardize_data(X_set):

    # Get mean of the set along the x-axis (horizontal sum)
    feature_mean = np.mean(X_set, axis=1, keepdims=1)

    # Subtract mean to get zero mean set
    X_zero_mean = X_set - feature_mean

    # compute variance along the x-axis (horizantal computation)
    feature_variance = np.std(X_set, axis=1, keepdims=1)

    # Compute standardized set
    X_standardized = X_zero_mean / feature_variance

    #  Extract values to print
    processed_mean = np.mean(X_standardized)
    processed_min = np.min(X_standardized)
    processed_max = np.max(X_standardized)

    # Print informations
    print(
        f"Standardized data - mean: ({processed_mean:.4f} - min: {processed_min:.4f} - max: {processed_max:.4f}"
    )

    # Return standardized set
    return X_standardized
