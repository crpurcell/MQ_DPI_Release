#=============================================================================#
#                                                                             #
# MODIFIED: 25-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import numpy as np

#-----------------------------------------------------------------------------#
class Perceptron:
    def __init__(self, N, alpha=0.1):
        """Initialize normalised weights and store the learning rate"""

        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        """Step function activation"""

        return 1 if x > 0 else 0

    def fit(self, X, y, epochs=10):
        """Perform a weight upate"""

        # Bias trick: stack a column of ones into the feature matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over the epochs & target data
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):

                # Perform the weighted sum
                p = self.step(np.dot(x, self.W))

                # Update weights if prediction is incorrect
                if p != target:
                    error = p -target
                    self.W += -self.alpha * error * x

    def predict(self, X, addBias=True):

        X = np.atleast_2d(X)   # Input must be a matrix (>1D)

        # Add bias column, if necessary
        if addBias:
            X = np.c_[X, np.ones((X.shape[0]))]

        # Dot product W . X  and pass through activation function
        return self.step(np.dot(X, self.W))
