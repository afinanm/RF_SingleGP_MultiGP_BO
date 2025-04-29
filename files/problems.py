from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

import numpy as np
import torch

# VERSION WITH STATIFIED K FOLD CV and

# Define the optimization problem
class RFOptimizationProblem:
    def __init__(self, seed, bounds, X_train, X_test, y_train, y_test):
        self.seed = seed
        self.bounds = bounds  # Define search space
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.dim = bounds.shape[1]
        self.num_objectives = bounds.shape[0] #2 for us always

        max_depth_bound = self.bounds[1,1]

        self.ref_point = torch.tensor([0, -1 * max_depth_bound + 2])  # Reference point for hypervolume, assume that it is the lower bound of the objective functions


    def evaluate_rf(self, x):
        """
        Trains an RF model for a row x and returns accuracy and tree depth.
        x: Tensor containing a set of hyperparameters, shape (1, 4)

        Returns:
            - Accuracy (to maximize)
            - Depth (to minimize)
        """

        n_estimators = int(x[0].item())
        max_depth = int(x[1].item()) if x[1].item() > 0 else None
        min_samples_split = int(x[2].item())
        min_samples_leaf = int(x[3].item())

        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.seed
        )
        #model.fit(self.X_train, self.y_train)
        # If we do cv we dont want to fit, we need to fit with the cross val score
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)

        # Perform cross-validation with the defined splitter
        cv_acc = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')

        accuracy = np.mean(cv_acc) - np.std(cv_acc)

        results = [accuracy, max_depth]

        return torch.tensor(results)  # Return tensor with shape (batch_size, 2)


    def evaluate(self, x):
        """
        Given input hyperparameters `x`, evaluate the RF model.
        Returns [accuracy, depth] (accuracy to be maximized, depth to be minimized).
        """
        results = self.evaluate_rf(x)  
        results[1] *= -1
        return results # Negate depth for minimization



