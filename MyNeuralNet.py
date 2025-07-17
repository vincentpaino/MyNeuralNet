import random
import numpy as np
import matplotlib.pyplot as plt

# what do we want our NN to predict?

class NeuralNetwork():
    def __init__(self, numHiddenLayers, numNeurons, batches=0):
        self.numHiddenLayers = numHiddenLayers
        self.numNeurons = numNeurons
        self.layers = np.zeros((numHiddenLayers, numNeurons))
        self.batches = batches
        self.inputs = [1,2,3,4,5,6,7,8,9,10]
        self.outputs = []
        self.weights = []
    
    def weightInitialization(self):
        for layer in self.numHiddenLayers:
            for neuron in self.numNeurons:

    def neuronInitilization(self):
        

    def forward(self):
        for _ in range(self.)
            self.inputs[_]

    def calculateGradients(self):
        gradients = []
        for layer in self.layers:
            for neuron in layer:
                # Placeholder for gradient calculation
                gradient = random.random()


    def train(self, epochs):
        self.forward(index)


    def activationFunction(self, neuron):
        # ReLU activation function
        return max(0,neuron)


    

if __name__ == "__main__":
    nn = NeuralNetwork(3, 5)
    print(f"Neural Network initialized with {nn.numLayers} layers and {nn.numNeurons}neurons per layer.")
    print(f"Layers structure: {nn.layers}")
    # Add more functionality as needed
    nn.weightInitialization()
    nn.