import numpy as np

from utils import *

class Layer:
    def __init__(self, input_size, layer_size):
        self.weights = np.random.uniform(-1, 1, (input_size, layer_size))
        self.bias = np.random.uniform(-1, 1, (1, layer_size))
        self.input = None
        self.output = None
        self.delta = None

    def forward(self, input):
        self.input = input
        self.output = sigmoid(np.dot(input, self.weights) + self.bias)
        return self.output

    def backward(self, error, learning_rate):
        self.delta = error * derivative_sigmoid(self.output)
        if self.input is not None:
            self.weights += learning_rate * np.dot(self.input.T, self.delta)
            self.bias += learning_rate * np.sum(self.delta, axis=0, keepdims=True)
        return np.dot(self.delta, self.weights.T)