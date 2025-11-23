from node import Node
import geometric_kinematics as gk
import sys
import math
import pickle
import random
import numpy as np

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

def log(log_file, average_angle_error, average_position_error, catastrophic_failures):
    with open(log_file, "a") as f:
        f.write(f"{average_angle_error},{average_position_error},{catastrophic_failures}\n")
    print(f"Avg Angle Error: {average_angle_error}, Avg Position Error: {average_position_error}, Catastrophic Failures: {catastrophic_failures}")
        
def load_network(pickle_file):
    with open(pickle_file, "rb") as f:
        return pickle.load(f)

def dump_network(pickle_file, nodes):
    network = []
    for layer in nodes:
        arr = []
        for node in layer:
            arr.append((node.weights, node.bias, node.layer, node.node_number))
        network.append(arr)
    with open(pickle_file, "wb") as f:
        pickle.dump(network, f)

def random_pair():
    angles = [random.uniform(gk.jointLowerLimits[i], gk.jointUpperLimits[i]) for i in range(len(gk.jointLowerLimits))]
    position = list(gk.Mat2Pose(gk.forwardKinematics(np.array(angles))))
    scaled_angles = [(angles[i] - gk.jointLowerLimits[i]) / (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) for i in range(len(angles))]
    return position, angles, scaled_angles

def predict(nodes: list[list[Node]], inputs):
    for layer in nodes:
        out = [node.output(inputs) for node in layer]
    return out

def update_params(nodes, expected, actual, loss_derivative, inputs):
    loss_derivatives = [loss_derivative(expected[i], actual[i]) for i in range(len(expected))]
    for layer in reversed(nodes):
        for node in layer:
            node.update(loss_derivatives, inputs)

if __name__ == "__main__":
    # Syntax = python neural_net.py <hidden layers> <activation_func_number> <loss_func_number> <training rounds> <load?> <save?>

    # Set up parameters
    hidden_layers = 1
    activation_func_number = 0
    loss_func_number = 0
    rounds = 100
    load = True
    save = True
    if len(sys.argv) > 1:
        hidden_layers = int(sys.argv[1])
        activation_func_number = int(sys.argv[2])
        loss_func_number = int(sys.argv[3])
        rounds = int(sys.argv[4])
        load = sys.argv[5].lower() == 'true'
        save = sys.argv[6].lower() == 'true'

    pickle_file = f"nn_h{hidden_layers}_o{activation_func_number}_l{loss_func_number}"
    log_file = f"nn_log_h{hidden_layers}_o{activation_func_number}_l{loss_func_number}.txt"

    if load: # Retrieve from file
        with open(pickle_file, "rb") as f:
            network_data = pickle.load(f)
        nodes = []
        for layer in network_data:
            arr = []
            for node_data in layer:
                weights, bias, layer_number, node_number = node_data
                node = Node(activation_functions[activation_func_number], derivatives[activation_func_number], input_vector_size, nodes, layer_number, node_number)
                node.weights = weights
                node.bias = bias
                arr.append(node)
            nodes.append(arr)
    else: # Create new network
        nodes = []

        for layer in range(hidden_layers + 1):
            arr = []
            
            if layer == hidden_layers:
                size = output_vector_size
            else:
                size = input_vector_size
            
            for node_number in range(size):
                arr.append(Node(activation_functions[activation_func_number], derivatives[activation_func_number], input_vector_size, nodes, layer, node_number))
            nodes.append(arr)

    training_rounds = 0
    total_angle_error = 0.0
    total_position_error = 0.0
    catastrophic_failures = 0

    # TRAINING LOOP
    try:
        for r in range(rounds):
            training_rounds += 1
            # scaled are between 0 and 1
            position, expected_angles, expected_scaled_angles = random_pair()
            predicted_angles = predict(nodes, position)

            for layer in nodes:
                for node in layer:
                    node.setup_gradients()

            unscaled_predicted_angles = [predicted_angles[i] * (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) + gk.jointLowerLimits[i] for i in range(len(predicted_angles))]
            total_angle_error += sum([abs(expected_angles[i] - unscaled_predicted_angles[i]) for i in range(len(expected_angles))])

            predicted_position = gk.Mat2Pose(gk.forwardKinematics(unscaled_predicted_angles))
            total_position_error += sum([abs(position[i] - predicted_position[i]) for i in range(len(position))])

            if is_catastrophic_failure(unscaled_predicted_angles):
                catastrophic_failures += 1

            update_params(nodes, expected_scaled_angles, predicted_angles, loss_derivatives[loss_func_number], position)

            if training_rounds % 100 == 0:
                # Log results and save network
                if save:
                    dump_network(pickle_file, nodes)

                average_angle_error = total_angle_error / training_rounds / 5
                average_position_error = total_position_error / training_rounds
                log(log_file, average_angle_error, average_position_error, catastrophic_failures)

                training_rounds = 0
                average_angle_error = 0.0
                average_position_error = 0.0   
                catastrophic_failures = 0
                
    except Exception as e: 
        print("FAILURE")
        dump_network(pickle_file, nodes)
        raise e
