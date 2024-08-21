import numpy as np

from utils import *

class MultilayerPerceptron:
    def __init__(self, layers, learning_rate=0.5):
        self.learning_rate = learning_rate
        self.layers = layers

    def train(self, input_data, output_data, epochs=1000):
        for epoch in range(epochs):
            
            prediction = self.predict(input_data)

            error = output_data - prediction
            
            error = self.layers[-1].backward(error, self.learning_rate)
            for layer in reversed(self.layers[:-1]):
                error = layer.backward(error, self.learning_rate)
            
            # Print loss cada 1000 épocas
            if (epoch + 1) % 1000 == 0:
                loss = mean_squared_error(output_data - prediction)
                print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f}')

    def predict(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data