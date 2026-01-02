import numpy as np
from InfoGain import gini, entropy, misclassification_error, chi_square

class Node:
    """
    Initialize a Node object with the given parameters.

    Parameters
    ----------
    feature : int
        The index of the feature to split on.
    threshold : float
        The value of the feature to split on.
    value : float
        The prediction value of the node.
    left : Node
        The left child of the node.
    right : Node
        The right child of the node.
    """
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        """
        Check if the node is a leaf node.

        Returns
        -------
        bool
            True if the node is a leaf node, False otherwise.
        """
        return self.value is not None


class DecisionTree:
    def __init__(self, criterion, max_depth, min_sample_split, alpha, max_features=None):
        """
        Initialize a DecisionTree object with the given parameters.

        Parameters
        ----------
        criterion : str
            The criterion to use when splitting nodes. Can be either "gini", "entropy", or "misclassification".
        max_depth : int
            The maximum depth of the tree.
        min_sample_split : int
            The minimum number of samples required to split an internal node.
        alpha : float
            The significance level to use when splitting nodes.
        max_features : int or str
            The maximum number of features to consider when splitting nodes. Can be either an integer or "sqrt".

        Returns
        -------
        None
        """
        self.root = None
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.alpha = alpha
        self.max_features = max_features

    def impurity(self, y):
        """
        Calculate the impurity of the given data according to the specified criterion.

        Parameters
        ----------
        y : array-like
            The labels of the data.

        Returns
        -------
        float
            The impurity of the given data.

        Raises
        -------
        ValueError
            If the specified criterion is unknown.

        Notes
        -----
        The impurity is calculated according to the specified criterion, which can be either "gini", "entropy", or "misclassification".
        """
        if self.criterion == "gini":
            return gini(y)
        elif self.criterion == "entropy":
            return entropy(y)
        elif self.criterion == "misclassification":
            return misclassification_error(y)
        else:
            raise ValueError("Unknown criterion")

    def information_gain(self, feature, y, threshold):
        """
        Calculate the information gain of the given feature and threshold.

        Parameters
        ----------
        feature : array-like
            The feature values.
        y : array-like
            The labels of the data.
        threshold : float
            The threshold value.

        Returns
        -------
        float
            The information gain of the given feature and threshold.

        Notes
        -----
        The information gain is calculated as the difference between the impurity of the parent node and the impurity of the child nodes.
        """
        parent_impurity = self.impurity(y)
        left_idx = feature <= threshold
        right_idx = feature > threshold

        if left_idx.sum() == 0 or right_idx.sum() == 0:
            return 0

        n = len(y)
        l, r = left_idx.sum(), right_idx.sum()
        left_impurity = self.impurity(y[left_idx])
        right_impurity = self.impurity(y[right_idx])

        child_impurity = (l * left_impurity + r * right_impurity) / n
        return parent_impurity - child_impurity

    def best_split(self, X, y, features):
        """
        Find the best feature and threshold to split the data.

        Parameters
        ----------
        X : array-like
            The feature values.
        y : array-like
            The labels of the data.
        features : array-like
            The features to consider.

        Returns
        -------
        best_feat : int
            The index of the best feature.
        best_thresh : float
            The best threshold value.
        best_gain : float
            The best information gain.
        """
        best_gain, best_feat, best_thresh = 0, None, None
        for feat in features:
            # get the unique values of the feature
            values = X[:, feat]
            thresholds = np.unique(values)

            # iterate over each unique value and calculate the information gain
            for t in thresholds:
                gain = self.information_gain(values, y, t)

                # if the gain is better than the best gain, update the best gain and its corresponding parameters
                if gain > best_gain:
                    best_gain, best_feat, best_thresh = gain, feat, t
        return best_feat, best_thresh, best_gain

    def grow(self, X, y, depth=0):
        print('growing tree at depth:', depth)
        """
        Recursively grow a decision tree.

        Parameters
        ----------
        X : array-like
            The feature values.
        y : array-like
            The labels of the data.
        depth : int
            The current depth of the tree.

        Returns
        -------
        Node
            The root of the decision tree.

        Notes
        -----
        The decision tree is recursively grown by finding the best feature and threshold to split the data, and then recursively calling the grow method on the left and right child nodes.
        """
        n_samples, n_features = X.shape

        # handle max_features
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                n_feats = int(np.sqrt(n_features))
            elif self.max_features == "log2":
                n_feats = int(np.log2(n_features))
            else:
                n_feats = n_features
        elif isinstance(self.max_features, int):
            n_feats = min(self.max_features, n_features)
        else:
            n_feats = n_features

        features = np.random.choice(n_features, n_feats, replace=False)

        # stopping condition
        if depth >= self.max_depth or n_samples < self.min_sample_split:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        feat, thresh, gain = self.best_split(X, y, features)
        if gain == 0:
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        left_idx, right_idx = X[:, feat] <= thresh, X[:, feat] > thresh

        # check for statistical significance
        if not chi_square(y, y[left_idx], y[right_idx], self.alpha):
            leaf_value = np.bincount(y).argmax()
            return Node(value=leaf_value)

        left = self.grow(X[left_idx], y[left_idx], depth + 1)
        right = self.grow(X[right_idx], y[right_idx], depth + 1)
        return Node(feature=feat, threshold=thresh, left=left, right=right)

    def fit(self, X, y):
        """
        Train a decision tree using the given data.

        Parameters
        ----------
        X : array-like
            The feature values.
        y : array-like
            The labels of the data.

        Returns
        -------
        None
        """
        self.root = self.grow(X, y)

    def predict_one(self, x, node):
        """
        Predict the label of a single data point using the decision tree.

        Parameters
        ----------
        x : array-like
            The feature values of the data point.
        node : Node
            The current node in the decision tree.

        Returns
        -------
        int
            The predicted label of the data point.
        """
        # Check if the node is a leaf node
        if node.is_leaf():
            # Return the predicted value if the node is a leaf node
            return node.value

        # Check if the feature value is less than or equal to the threshold
        if x[node.feature] <= node.threshold:
            # Recursively call the predict_one method on the left child node
            return self.predict_one(x, node.left)
        else:
            # Recursively call the predict_one method on the right child node
            return self.predict_one(x, node.right)

    def predict(self, X):
        """
        Predict the labels of the given data points using the decision tree.

        Parameters
        ----------
        X : array-like
            The feature values of the data points.

        Returns
        -------
        array-like
            The predicted labels of the data points.
        """
        # Predict the labels of the given data points using the decision tree
        return np.array([self.predict_one(x, self.root) for x in X])
 