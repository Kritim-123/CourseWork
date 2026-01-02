import numpy as np
from scipy.stats import chi2

def entropy(y):
    """
    Calculate the entropy of the given labels.

    Parameters
    ----------
    y : array-like
        The labels to calculate the entropy for.

    Returns
    -------
    float
        The entropy of the given labels.
    """
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    # add a small value to avoid log2(0)
    return -np.sum(probs * np.log2(probs + 1e-9))

def gini(y):
    """
    Calculate the Gini impurity of the given labels.

    Parameters
    ----------
    y : array-like
        The labels to calculate the Gini impurity for.

    Returns
    -------
    float
        The Gini impurity of the given labels.
    """
    labels, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)

    # calculate the Gini impurity
    gini_impurity = 1 - np.sum(probs ** 2)
    return gini_impurity

def misclassification_error(y):
    """
    Calculate the misclassification error of the given labels.

    Parameters
    ----------
    y : array-like
        The labels to calculate the misclassification error for.

    Returns
    -------
    float
        The misclassification error of the given labels.
    """
    # get the unique labels and their counts
    labels, counts = np.unique(y, return_counts=True)

    # calculate the probabilities of each label
    probs = counts / len(y)

    # calculate the misclassification error
    return 1 - np.max(probs)

def chi_square(parent, left, right, alpha=0.05):
    """
    Perform the chi-squared test to determine if the given data is independent.

    Parameters
    ----------
    parent : array-like
        The parent data.
    left : array-like
        The left child data.
    right : array-like
        The right child data.
    alpha : float, optional
        The significance level to use. Defaults to 0.05.

    Returns
    -------
    bool
        True if the data is independent, False otherwise.
    """
    count_zero_parent = np.count_nonzero(parent == 0)
    count_one_parent = np.count_nonzero(parent == 1)

    count_zero_left = np.count_nonzero(left == 0)
    count_one_left = np.count_nonzero(left == 1)

    count_zero_right = np.count_nonzero(right == 0)
    count_one_right = np.count_nonzero(right == 1)

    # Calculate the probabilities of the parent data
    prob_zero = count_zero_parent / len(parent)
    prob_one = count_one_parent / len(parent)

    # Calculate the expected values of the left and right child data
    expected_zero_left = prob_zero * len(left)
    expected_one_left = prob_one * len(left)
    expected_zero_right = prob_zero * len(right)
    expected_one_right = prob_one * len(right)

    # Calculate the observed values of the left and right child data
    observed = np.array([count_zero_left, count_one_left,
                             count_zero_right, count_one_right])

    # Calculate the expected values of the left and right child data
    expected = np.array([expected_zero_left, expected_one_left,
                             expected_zero_right, expected_one_right]) + 1e-9

    # Calculate the chi-squared statistic
    chi_stat = np.sum((observed - expected) ** 2 / expected)

    # Calculate the p-value
    p_value = 1 - chi2.cdf(chi_stat, df=1)

    # Return True if the data is independent, False otherwise
    return p_value < alpha
