import numpy as np

# Função de ativação Sigmoid e sua derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Função de custo (erro quadrático médio)
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Classe da Rede Neural com Backpropagation
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        # Inicializa pesos aleatórios
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.bias_hidden = np.random.uniform(-1, 1, (1, hidden_size))
        self.bias_output = np.random.uniform(-1, 1, (1, output_size))

    def forward(self, X):
        # Passagem direta (feedforward)
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = sigmoid(self.hidden_input)
        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y_true, y_pred):
        # Erro na saída
        error_output_layer = y_true - y_pred
        delta_output = error_output_layer * sigmoid_derivative(y_pred)
        
        # Erro na camada oculta
        error_hidden_layer = np.dot(delta_output, self.weights_hidden_output.T)
        delta_hidden = error_hidden_layer * sigmoid_derivative(self.hidden_output)
        
        # Atualização de pesos e bias
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden_output.T, delta_output)
        self.bias_output += self.learning_rate * np.sum(delta_output, axis=0, keepdims=True)
        self.weights_input_hidden += self.learning_rate * np.dot(X.T, delta_hidden)
        self.bias_hidden += self.learning_rate * np.sum(delta_hidden, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            # Passo de feedforward
            y_pred = self.forward(X)
            # Passo de backpropagation
            self.backward(X, y, y_pred)
            # Mostrar o erro a cada 1000 épocas
            if (epoch+1) % 1000 == 0:
                loss = mean_squared_error(y, y_pred)
                print(f'Epoch {epoch+1}/{epochs} - Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

# Dados para o problema XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Inicializar a rede neural MLP
mlp = MLP(input_size=2, hidden_size=2, output_size=1, learning_rate=0.5)

# Treinar a rede por 10.000 épocas
mlp.train(X, y, epochs=10000)

# Testar a rede com as entradas XOR
for input_data, output_data in zip(X, y):
    predicted = mlp.predict(input_data.reshape(1, -1))
    print(f'Input: {input_data}, Predicted: {predicted}, Actual: {output_data}')
