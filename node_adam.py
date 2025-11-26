import random
import math

learning_rate = 1
beta_1 = 0.95
beta_2 = 0.95
epsilon = 0.00000001

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
        self.moments = [0.0 for _ in range(len(self.weights))]
        self.moments_corrected = [0.0 for _ in range(len(self.weights))]
        self.second_moment = [0.0 for _ in range(len(self.weights))]
        self.second_moment_corrected = [0.0 for _ in range(len(self.weights))]
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.bias_moment = 0.0
        self.bias_moment_corrected = 0.0
        self.second_bias_moment = 0.0
        self.second_bias_moment_corrected = 0.0

        

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
        for i in range(len(self.weights)):
            if self.layer == 0:
                self.gradient_values[i] = self.gradients[i](loss_derivatives, inputs)
            else:
                self.gradient_values[i] = self.gradients[i](loss_derivatives)

            self.moments[i] = beta_1 * self.moments[i] + (1 - beta_1) * self.gradient_values[i]
            self.second_moment[i] = beta_2 * self.second_moment[i] + (1 - beta_2) * self.gradient_values[i] ** 2
            
            self.moments_corrected[i] = self.moments[i] / (1 - self.beta_1)
            self.second_moment_corrected[i] = self.second_moment[i] / (1 - self.beta_2)

            self.weights[i] += learning_rate * self.moments_corrected[i] / math.sqrt(self.second_moment_corrected[i] + epsilon)

        self.bias_gradient_value = self.bias_gradient(loss_derivatives)
        self.bias_moment = beta_1 * self.bias_moment + (1 - beta_1) * self.bias_gradient_value
        self.second_bias_moment = beta_2 * self.second_bias_moment + (1 - beta_2) * self.bias_gradient_value ** 2

        self.bias_moment_corrected = self.bias_moment[i] / (1 - self.beta_1)
        self.second_bias_moment_corrected = self.bias_moment[i] / (1 - self.beta_2)

        self.bias += learning_rate * self.bias_moment_corrected / math.sqrt(self.second_bias_moment_corrected + epsilon)

        self.beta_1 *= beta_1
        self.beta_2 *= beta_2