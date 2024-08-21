import numpy as np

from Layer import Layer
from MultilayeredPercepetron import MultilayerPerceptron

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
