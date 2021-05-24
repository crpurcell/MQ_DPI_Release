#=============================================================================#
#                                                                             #
# MODIFIED: 25-Jun-2018 by C. Purcell                                         #
#                                                                             #
#=============================================================================#

import numpy as np

#-----------------------------------------------------------------------------#
class NeuralNetwork:
    def __init__(self, layers, alpha=0.1):
        self.W = []                  # List of weight matrices
        self.layers = layers         # Architecture as a list (e.g., [2,2,1])
        self.alpha = alpha           # Learning rate

        # Loop through the layers up until the last two
        for i in np.arange(0, len(layers) - 2):

            # Randomly initialise normalised weights, including a bias.
            # Connecting M x N, where M=layers[i] & N = layers[i + 1].
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))

        # The last two layers are a special case where the input
        # connections need a bias term but the output does not
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        """Python magic method that is called by print(NeuralNetwork).
        Here we just return the architecture as a string."""
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers))

    def sigmoid(self, x):
        """Evaluate a signmoid activation function"""
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        """Evaluate the derivative of the sigmoid (assuming x holds result)"""
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, displayUpdate=100):
        """Function to train the neural network"""

        # Implement bias trick
        X = np.c_[X, np.ones((X.shape[0]))]

        # Loop over epochs and data
        for epoch in np.arange(0, epochs):
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)

            # Feedback: print the loss
            if epoch == 0 or (epoch + 1) % displayUpdate == 0:
                loss = self.calculate_loss(X, y)
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1, loss))

    def fit_partial(self, x, y):
        """Perform back-probagation to fit the NN"""

        # List to store output activations for each layer
        # Initialised to the input feature vector
        A = [np.atleast_2d(x)]

        # FEEDFORWARD: loop over the layers in the network
        for layer in np.arange(0, len(self.W)):

            # Calculate the "net input" to the current layer
            net = A[layer].dot(self.W[layer])

            # Append "net output", = activation(net_input)
            out = self.sigmoid(net)
            A.append(out)

        # BACKPROPAGATION
        # the first phase of backpropagation is to compute the
        # difference between our *prediction* (the final output
        # activation in the activations list) and the true target
        # value
        error = A[-1] - y

        # from here, we need to apply the chain rule and build our
        # list of deltas 'D'; the first entry in the deltas is
        # simply the error of the output layer times the derivative
        # of our activation function for the output value
        D = [error * self.sigmoid_deriv(A[-1])]

        # once you understand the chain rule it becomes super easy
        # to implement with a for loop -- simply loop over the
        # layers in reverse order (ignoring the last two since we
        # already have taken them into account)
        for layer in np.arange(len(A) - 2, 0, -1):
            # the delta for the current layer is equal to the delta
            # of the *previous layer* dotted with the weight matrix
            # of the current layer, followed by multiplying the delta
            # by the derivative of the non-linear activation function
            # for the activations of the current layer
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)

        # Reverse the deltas
        D = D[::-1]

        # WEIGHT UPDATE PHASE
        for layer in np.arange(0, len(self.W)):
            # update our weights by taking the dot product of the layer
            # activations with their respective deltas, then multiplying
            # this value by some small learning rate and adding to our
            # weight matrix -- this is where the actual "learning" takes
            # place
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):

        # Initialize the output prediction as the input features
        p = np.atleast_2d(X)

        # Add bias column, if necessary
        if addBias:
            p = np.c_[p, np.ones((p.shape[0]))]

        # Propagate input through the network
        for layer in np.arange(0, len(self.W)):
            p = self.sigmoid(np.dot(p, self.W[layer]))

        return p

    def calculate_loss(self, X, targets):
        """Make predictions for the input data points then compute the loss"""

        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)

        return loss
