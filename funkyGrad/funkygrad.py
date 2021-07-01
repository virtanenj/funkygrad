import numpy as np


# Simple feedforward neural network:
#
# x (input layer) -> hidden layers -> y (output layer) -> backpropagation -> ...
#
# Forward pass:
# a_i = sigmoid(w_i.a_{i-1} + b_i)
# => a = (a_1, ..., a_m)
#
# Backpropagation:
# - updates weights and biases by trying to find the minimum of the
# loss function (i.e. stochastic gradient descent)
# 
# Note: use automatic differentiation instead of symbolic differentiation. (How???)


class FNN():
    '''
    How is this used:
    1. Initialize the neural network
        model = fnn(input_size, output_size, hidden_layer_size, activation_function)
        model.initialize_weights_and_biases()
    2. Set the loss funciton and optimizer and compile
    3. Training
    4. Prediction
    '''
    def __init__(self, input_size, output_size, hidden_layer_sizes, activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation_function = activation_function
        self.Ws = None
        self.Bs = None
        self.loss_function = None
        self.optimizer = None

    def initialize_weights_and_biases(self):
        '''
        Set the weights and biases to some random values
        '''
        Ws = []
        Bs = []
        # Mapping from input layer into  the 1. hidden layer
        W = np.random.normal(size=(self.hidden_layer_sizes[0], self.input_size))
        B = np.random.normal(size=self.hidden_layer_sizes[0])
        Ws.append(W)
        Bs.append(B)
        for i in range(1, len(self.hidden_layer_sizes)):
            # Mapping from i-1. to i. hidden layer
            W = np.random.normal(size=(self.hidden_layer_sizes[i], self.hidden_layer_sizes[i-1]))
            B = np.random.normal(size=self.hidden_layer_sizes[i])
            Ws.append(W)
            Bs.append(B)
        # Mapping from the last hidden layer into the output layer
        W = np.random.normal(size=(self.output_size, self.hidden_layer_sizes[-1]))
        B = np.random.normal(size=self.output_size)
        Ws.append(W)
        Bs.append(B)
        self.Ws = Ws
        self.Bs = Bs

    def forward_propagation(self, x):
        '''
        Calculation from the input layer to the output layer using the currect 
        weights and biases.
        '''
        # Input layer:
        a = x.copy()
        # Hidden layers + output layer:
        for i in range(len(self.hidden_layer_sizes) + 1):
            # Error: ?????
            Wi = self.Ws[i]
            Bi = self.Bs[i]
            a = self.activation_function(Wi.dot(a) + Bi)  # == activation_function(matmul(Wi, a) + Bi)
        return a

    # # Should this particular loss function and gradient descent for it 
    # # be done in a separate class?
    # def loss_function(self, y_true, y_pred):
    #     '''
    #     Mean squarred error (MSE) = (1/n) sum_{i=1}^n (y_true - y_pred)^2
    #     where n is the number of observartion. 
    #     '''

    #     # sum over all of (y_pred - y_true)**2
        
    #     pass

    # def gradient_descent(self):
    #     '''
    #     -grad(loss_function(Ws, Bs)) = -grad(C(Ws, Bs))
    #     = - dC/dw1 - dC/dw2 - ... - dC/dB1 - ...

    #     where each weight and bias of each layer are summed.
    #     '''
    #     # Can this be done more simply and quickly, using Numpy for example?
    #     gradient = []
    #     for W in Ws:
    #         for w in W:
    #             # Do the gradient with w
    #     for B in Bs:
    #         for b in B:
    #             # do the gradient with b
    #     return gradient

    def backpropagation(self):
        '''
        Updates the weighs and biases 
        '''

        pass

    def compile(self):
        pass

    def training(self, epochs):
        '''
        forwar propagation + backpropagtion
        '''
        pass

# Some (common) loss function
class MeanAbsoluteError:
    '''
    The loss/cost function:
        L = sum_{i=1}^n (y_pred_i - y_true_i)^2
    and its 

    '''
    def __init__(self):
        pass

    def loss_function(self, y_pred, y_true):
        return np.sum((y_pred - y_true)**2)

    def gradient_descent(self):
        pass



# Some (common) activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))





# input_size = 5
# hidden_layer_sizes = [10, 12]
# output_size = 2


# # Some random input:
# x = np.random.random(input_size)

# tester = FNN(input_size, output_size, hidden_layer_sizes, sigmoid)
# tester.initialize_weights_and_biases()
# output = tester.forward_propagation(x)

