import random

import numpy as np
from tqdm import tqdm

from activation_functions import sigmoid, sigmoid_prime
from cost_functions import mse_derivative
 
 
class Network:
    # sizes is a list of the number of nodes in each layer
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x,y in zip(sizes[:-1], sizes[1:])]
       
    def feedforward(self, z):
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, z) + b
            z = sigmoid(z)
        return z
   
    def train(self, training_data, epochs, mini_batch_size, lr, test_data=None):
        training_data = list(training_data)
        samples = len(training_data)
        
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)
       
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size]
                            for k in range(0, samples, mini_batch_size)]
            for mini_batch in tqdm(mini_batches):
                self.update_mini_batch(mini_batch, lr)
            if test_data:
                print(f"Epoch {j}: {self.predict(test_data)}/ {n_test}")
            else:
                print(f"Epoch {j} complete")
   
    def backprop(self, x, y):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        # forward
        activation = x
        activations = [x] # stores activations layer by layer
        zs = [] # stores z vectors layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
       
        # backward pass
        delta = mse_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        new_b[-1] = delta
        new_w[-1] = np.dot(delta, activations[-2].transpose())
       
        for _layer in range(2, self.num_layers):
            z = zs[-_layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-_layer+1].transpose(), delta) * sp
            new_b[-_layer] = delta
            new_w[-_layer] = np.dot(delta, activations[-_layer-1].transpose())
        return (new_b, new_w)
   
    def update_mini_batch(self, mini_batch, lr):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_new_b, delta_new_w = self.backprop(x, y)
            new_b = [nb + dnb for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [nw + dnw for nw, dnw in zip(new_w, delta_new_w)]
        self.weights = [w-(lr/len(mini_batch))*nw
                        for w, nw in zip(self.weights, new_w)]
        self.biases = [b-(lr/len(mini_batch))*nb
                       for b, nb in zip(self.biases, new_b)]
       
    def predict(self, test_data):
        all_res = []
        for d in test_data:
            z = self.feedforward(d[0])
            results = (np.argmax(z), d[1])
            all_res.append(results)
        return sum(int(y[x]) for (x, y) in all_res)
       
            
