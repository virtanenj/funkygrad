# Disables some warnings (TensorFlow gives a lot of them)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # <- 0, 1, 2 or 3
from tensorflow.keras.datasets import mnist  # Examples to test FunkyGrad
import numpy as np
import funkygrad
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = mnist.load_data()

input_size = 28*28
output_size = 10
hidden_layer_sizes = [64, 64]

model = funkygrad.FNN(input_size, output_size, hidden_layer_sizes, funkygrad.sigmoid)
model.initialize_weights_and_biases()
