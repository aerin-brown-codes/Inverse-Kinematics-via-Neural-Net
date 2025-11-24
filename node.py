import random

learning_rate = 0.1

class Node:
    def __init__(self, activation_function, activation_derivative, input_size: int, nodes: list[list], layer: int, node_number: int):
        self.weights = [random.random() for _ in range(input_size)]
        self.bias = 0.0
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative
        self.value = None
        self.nodes = nodes
        self.layer = layer
        self.node_number = node_number
        self.gradients = []
        self.bias_gradient = None
        self.gradient_values = [0.0 for _ in range(len(self.weights))]
        self.bias_gradient_value = 0.0

    def setup_gradients(self):
        if self.layer == len(self.nodes) - 1: # is last row
            self.gradients = [lambda loss_derivatives: loss_derivatives[self.node_number] * self.activation_derivative(self.value) * self.nodes[self.layer - 1][i].value for i in range(len(self.weights))] 
            self.bias_gradient = lambda loss_derivatives: loss_derivatives[self.node_number]
        elif self.layer == 0: # is first row
            self.gradients = [lambda loss_derivatives, inputs: sum([node.gradient_values[i] for node in self.nodes[self.layer + 1]]) * self.activation_derivative(self.value) * inputs[i] for i in range(len(self.weights))]
            self.bias_gradient = lambda loss_derivatives: sum([node.bias_gradient_value for node in self.nodes[self.layer + 1]]) * self.activation_derivative(self.value)
        else:
            self.gradients = [lambda loss_derivatives: sum([node.gradient_values[i] for node in self.nodes[self.layer + 1]]) * self.activation_derivative(self.value) * self.nodes[self.layer - 1][i].value for i in range(len(self.weights))]
            self.bias_gradient = lambda loss_derivatives: sum([node.bias_gradient_value for node in self.nodes[self.layer + 1]]) * self.activation_derivative(self.value)


    def output(self, inputs: list | None) -> float:
        if self.layer == 0:
            self.value = self.activation_function(sum([w * i for w, i in zip(self.weights, inputs)]) + self.bias)
        else:
            self.value = self.activation_function(sum([self.weights[i] * self.nodes[self.layer - 1][i].value for i in range(len(self.weights))]) + self.bias)
        return self.value

    def update(self, loss_derivatives = list, inputs: list | None = None):
        if self.layer == 0:
            for i in range(len(self.weights)):
                self.gradient_values[i] = self.gradients[i](loss_derivatives, inputs)
                self.weights[i] += learning_rate * self.gradient_values[i]
        else:
            for i in range(len(self.weights)):
                self.gradient_values[i] = self.gradients[i](loss_derivatives)
                self.weights[i] += learning_rate * self.gradient_values[i]
        self.bias_gradient_value = self.bias_gradient(loss_derivatives)
        self.bias += learning_rate * self.bias_gradient_value