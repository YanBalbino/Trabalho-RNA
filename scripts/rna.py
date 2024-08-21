import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return x * (1 - x)

def mean_squared_error(error):
    return np.mean((error)**2)

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

def main():
    # 2 inputs, 3 hidden units, 1 output
    mlp = MultilayerPerceptron([Layer(2, 3), Layer(3, 1)], 0.5)
    
    input_data = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    output_data = np.array([
        [0],
        [1],
        [1],
        [0]
    ])

    mlp.train(input_data, output_data, 1500)

    # Testes
    for input_data_single, output_data_single in zip(input_data, output_data):
        predicted = mlp.predict(input_data_single.reshape(1, -1))
        print(f'Input: {input_data_single}, Predição: {predicted.flatten()}, Correto: {output_data_single}')

if __name__ == "__main__":
    main()
