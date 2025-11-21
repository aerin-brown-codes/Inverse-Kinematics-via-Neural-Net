from node import Node
import geometric_kinematics as gk
import sys
import math

input_vector_size = 6

activation_functions = [
    lambda a: 1 / (1 + math.exp(-a)), # Sigmoid
    lambda a: max(0, a), # ReLU
    lambda a: math.log(1 + math.exp(a)) # Softplus
]

loss_functions = [
    lambda h, y: [(h_i - y_i) ** 2 for h_i, y_i in zip(h, y)], # Squared loss
]

def is_catastrophic_failure(angles: list[float]) -> bool:
    for i in range(len(angles)):
        if angles[i] < gk.jointLowerLimits[i] or angles[i] > gk.jointUpperLimits[i]:
            return True
    return False

if __name__ == "__main__":
    # Syntax = python neural_net.py <hidden layers> <activation_func_number> <loss_func_number>

    # Set up parameters
    hidden_layers = 1
    activation_func_number = 0
    loss_func_number = 0
    if len(sys.argv) > 2:
        hidden_layers = int(sys.argv[2])
        activation_func_number = int(sys.argv[3])
        loss_func_number = int(sys.argv[4])

    filename = f"nn_h{hidden_layers}_o{activation_func_number}_l{loss_func_number}.txt"

    nodes = [[Node(activation_functions[activation_func_number]) for _ in range(input_vector_size)] for _ in range(hidden_layers) + 1]

