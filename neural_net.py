import numpy as np


# Simple feedforward neural network:
#
# x -> hidden layers -> y -> backpropagation -> ....
#
# Forward pass:
# a_i = sigmoid(w_i.a_{i-1} + b_i)
# => a = (a_1, ..., a_m)
#
# Backpropagation:
# - updates weights and biases by trying to find the minimum of the
# loss function, such as stochastic gradient descent


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


input_size = 5
hidden_layer_sizes = [10, 12]
hidden_layers = 2  # == len(layers) - 1
# -> layers = [5, 10, 10]
layers = [input_size]
for hidden_layer in hidden_layer_sizes:
    layers.append(hidden_layer)


np.random.seed(0)


# Input layer:
x = np.random.random(layers[0])


class fnn():
    def __init__(self, layers):
        self.layers = layers
        self.Ws = None
        self.bs = None

    def initialize_layers(self, layers):
        # ???
        pass

    def initialize_weights_and_biases(self):
        # Different ways of doing this???
        Ws = []
        Bs = []
        for i in range(len(self.layers) - 1):
            W = np.random.uniform((self.layers[i + 1], self.layers[i]))
            Ws.append(W)
            B = np.random.uniform(self.layers[i + 1])
            Bs.append(B)
        self.Ws = Ws
        self.bs = Bs

    def forward_propagation(self, x):
        hidden_layers = len(self.layers) - 1
        a = x.copy()
        for i in range(hidden_layers):
            Wi = self.Ws[i]
            Bi = self.bs[i]
            a = sigmoid(Wi.dot(a) + Bi)
        return a

    def backpropagation(self):
        pass
