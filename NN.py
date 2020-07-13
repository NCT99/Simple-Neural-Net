import math, random
import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_layers, output_dim, bias = False):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = 0.5
        self.layers = hidden_layers.copy()
        self.layers.append(output_dim)
        self.layers.insert(0, input_dim)
        self.bias = bias
        self.network = self.build_network()

    def train(self, data_in, data_out):
        for (x, y) in zip(data_in, data_out):
            self.forward_propagate(x)
            self.back_propagate(y)
            self.update_weights(x)

    def predict(self, x_in):
        y_out = []
        self.forward_propagate(x_in)
        for node in self.network[-1]:
            y_out.append(node['output'])
        return y_out

    #Internal functions

    def build_network(self):
        network = []

        def layer(in_dim, out_dim, last):
            layer = []
            if(self.bias):
                if(not last):
                    node = {'weights' : [], 'output' : 1, 'delta' : None}
                    layer.append(node)
                in_dim += 1
            for i in range(out_dim):
                weights = [random.random() for e in range(in_dim)]
                node = {'weights' : weights, 'output' : None, 'delta' : None}
                layer.append(node)
            return layer

        for i in range(1, len(self.layers)):
             network.append(layer(self.layers[i - 1], self.layers[i], last = (i == len(self.layers)-1)))
        return network

    def forward_propagate(self, x):
        x_in = x.copy()
        if(self.bias):
            x_in.insert(0, 1)
        for i in range(len(self.network)):
            x_out = []
            for node in self.network[i]:
                if len(node['weights']) != 0:
                    node['output'] = self.sigmoid(self.dot(x_in, node['weights']))
                x_out.append(node['output'])
            x_in = x_out

    def back_propagate(self, y_out):
        for i in reversed(range(len(self.network))):
            der = self.sigmoid_derivative
            if i == len(self.network)-1:
                for j, node in enumerate(self.network[-1]):
                    node['delta'] = 2 * (node['output'] - y_out[j]) * der(node['output'])
                                     
            else:
                for j, node in enumerate(self.network[i]):
                    s = 0
                    for out_node in self.network[i+1]:
                        if len(out_node['weights']) != 0:
                            s += out_node['weights'][j] * out_node['delta']
                    node['delta'] = der(node['output']) * s

    def update_weights(self, x):
        x_in = x.copy()
        if(self.bias):
            x_in.insert(0, 1)
        for i in range(len(self.network)):
            if i == 0:
                for node in self.network[i]:
                    for j in range(len(node['weights'])):
                        node['weights'][j] -= self.learning_rate * x_in[j] * node['delta']
            else:
                for node in self.network[i]:
                    for j in range(len(node['weights'])):
                        in_node = self.network[i-1][j]
                        node['weights'][j] -= self.learning_rate * in_node['output'] * node['delta']

    def sigmoid(self, x):
        return 1.0/(1.0 + math.exp(-x))

    def sigmoid_derivative(self, sigmoid):
         return sigmoid * (1 - sigmoid)

    def dot(self, a, b):
        return sum([a_ * b_ for (a_, b_) in zip(a, b)])
