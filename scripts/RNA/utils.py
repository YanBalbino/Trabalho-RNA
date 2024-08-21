import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

def mean_squared_error(error):
    return np.mean((error)**2)