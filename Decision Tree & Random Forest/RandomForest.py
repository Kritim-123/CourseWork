import numpy as np

from DecisionTree import DecisionTree

'''
Steps: 

1. Create a bootstrapped dataset

2. Create a decision tree using the bootstrapped dataset, but only use a random subset of variables 
(or columns) at each step.

3. Repeat Step 1 and 2

4. There will be some dataset that would not be listed in the bootstrapped data.(Out of bag samples)
We would use these out of bag samples through all of the other trees that were built without it.
'''

class RandomForest:
    def __init__(self, n_estimators, criterion, max_depth,
                 min_sample_split, alpha, max_features):
        self.n_estimators = n_estimators ## Number of trees we are going to use
        self.criterion = criterion ##Criterion we are using
        self.max_depth = max_depth ## Max depth of decision tree
        self.min_sample_split = min_sample_split ## Min number of element in a leaf in a decision tree
        self.alpha = alpha  ## alpha value for decision tree
        self.max_features = max_features ## Max number of features we are going to use
        self.trees = []

    def bootstrap(self, X, y):
        """
        Balanced bootstrap sampling: equal fraud vs non-fraud.
        
        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like
            The labels for the input data.
        
        Returns
        -------
        X_s, y_s : tuple of array-like
            Bootstrapped samples of X and y.
        """
        fraud_idx = np.where(y == 1)[0]
        legit_idx = np.where(y == 0)[0]

        n_samples = X.shape[0] // 2
        fraud_sample = np.random.choice(fraud_idx, n_samples // 2, replace=True)
        legit_sample = np.random.choice(legit_idx, n_samples // 2, replace=True)

        idxs = np.concatenate([fraud_sample, legit_sample])
        np.random.shuffle(idxs)
        return X[idxs], y[idxs]

    def fit(self, X, y):
        """
        Train a random forest using the bootstrapped dataset.

        Parameters
        ----------
        X : array-like
            The input data.
        y : array-like
            The labels for the input data.
        """
        self.trees = []
        print('before for loop')
        for _ in range(self.n_estimators):
            X_s, y_s = self.bootstrap(X, y)
            # Create a decision tree using the bootstrapped dataset
            # and only use a random subset of variables (columns) at each step (which is happening in decison Tree class)
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_sample_split=self.min_sample_split,
                alpha=self.alpha,
                max_features=self.max_features
            )
            # Train the decision tree using the bootstrapped dataset
            tree.fit(X_s, y_s)
            # Add the decision tree to the list of trees
            self.trees.append(tree)


    '''
    We will be taking mean value of the predictions of all the trees
    and if the vote is greater than 0.5 we will consider it as 1
    
    '''
    def predict_proba(self, X):
        preds = np.array([tree.predict(X) for tree in self.trees])
        return preds.mean(axis=0)

    def predict(self, X, threshold=0.5):
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)
