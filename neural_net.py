from node import Node
import geometric_kinematics as gk
import sys
import math
import pickle
import autograd.numpy as np
from autograd import grad

input_vector_size = 6 # (x, y, z, alpha, beta, gamma)
output_vector_size = 5 # (waist, shoulder, elbow, wrist, hand)

activation_functions = [
    lambda a: 1 / (1 + math.exp(-a)), # Sigmoid 
    lambda a: max(0, a), # ReLU
    lambda a: math.log(1 + math.exp(a)) # Softplus
]

derivatives = [
    lambda a: (a * (1 - a)), # Derivative of Sigmoid wrt value of sigmoid
]

loss_functions = [
    lambda h, y: [(h_i - y_i) ** 2 for h_i, y_i in zip(h, y)], # Squared loss function
    lambda h, y: [-(h_i * math.log(y_i) + (1 - h_i) * math.log(1 - y_i)) for h_i, y_i in zip(h, y)] # Cross-entropy loss function
]

loss_derivatives = [
    lambda expected, actual: -2 * (expected - actual) # Derivative of squared loss function for one value
]

def is_catastrophic_failure(angles: list[float]) -> bool:
    for i in range(len(angles)):
        if angles[i] < gk.jointLowerLimits[i] or angles[i] > gk.jointUpperLimits[i]:
            return True
    return False

def log(log_file, training_rounds, average_error, catastrophic_failures):
    with open(log_file, "a") as f:
        f.write(f"{training_rounds},{average_error},{catastrophic_failures}\n")
        
def load_network(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def dump_network(pickle_file, nodes):
    with open(pickle_file, "wb") as f:
        pickle.dump(nodes, f)

def predict(nodes: list[list[Node]], inputs):
    for layer in nodes:
        out = [node.output(inputs) for node in layer]
    return out

if __name__ == "__main__":
    # Syntax = python neural_net.py <hidden layers> <activation_func_number> <loss_func_number> <training rounds> <load?> <save?>

    # Set up parameters
    hidden_layers = 1
    activation_func_number = 0
    loss_func_number = 0
    rounds = 100
    load = True
    save = True
    if len(sys.argv) > 2:
        hidden_layers = int(sys.argv[2])
        activation_func_number = int(sys.argv[3])
        loss_func_number = int(sys.argv[4])
        rounds = int(sys.argv[5])
        load = sys.argv[6].lower() == 'true'
        save = sys.argv[7].lower() == 'true'

    pickle_file = f"nn_h{hidden_layers}_o{activation_func_number}_l{loss_func_number}"
    log_file = f"nn_log_h{hidden_layers}_o{activation_func_number}_l{loss_func_number}.txt"

    if load:
        nodes = load_network(pickle_file)
    else:
        nodes = []

        for layer in range(hidden_layers + 1):
            arr = []
            
            if layer == hidden_layers:
                size = output_vector_size
            else:
                size = input_vector_size
            
            for node_number in range(size):
                arr.append(Node(activation_functions[activation_func_number], derivatives[activation_func_number], input_vector_size, nodes, layer, node_number))

    training_rounds = 0
    average_error = 0.0
    catastrophic_failures = 0
