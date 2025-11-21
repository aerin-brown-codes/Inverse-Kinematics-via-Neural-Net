alpha = 0.1

class Node:
    def __init__(self, activation_function):
        self.weights = []
        self.bias = 0.0
        self.output_function = activation_function

    def output(self, inputs):
        return self.activation_function(sum(w * x for w, x in zip(self.weights, inputs)) + self.bias)

    def update(self, loss):
        pass