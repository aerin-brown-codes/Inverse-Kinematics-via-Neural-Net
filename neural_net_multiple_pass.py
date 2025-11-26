from node import Node
import geometric_kinematics as gk
import sys
import math
import pickle
import random
import numpy as np
import datetime

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
    lambda expected, actual: [(expected_i - actual_i) ** 2 for expected_i, actual_i in zip(expected, actual)], # Squared loss function
    lambda h, y: [-(expected_i * math.log(actual_i) + (1 - expected_i) * math.log(1 - actual_i)) for expected_i, actual_i in zip(expected, actual)] # Cross-entropy loss function
]

loss_derivatives = [
    lambda actual, expected: -2 * (expected - actual) # Derivative of squared loss function for one value
]

def is_catastrophic_failure(angles: list[float]) -> bool:
    for i in range(len(angles)):
        if angles[i] < gk.jointLowerLimits[i] or angles[i] > gk.jointUpperLimits[i]:
            return True
    return False

def log(log_file, average_angle_error, standard_deviation, average_position_error, catastrophic_failures, average_time):
    with open(log_file, "a") as f:
        f.write(f"{average_angle_error},{standard_deviation},{average_position_error},{catastrophic_failures},{average_time}\n")
    print(f"Avg Angle Error: {average_angle_error}, Standard Deviation: {standard_deviation}, Avg Position Error: {average_position_error}, Catastrophic Failures: {catastrophic_failures}")#, Avg Time: {average_time}")

def dump_network(pickle_file, nodes):
    network = []
    for layer in nodes:
        arr = []
        for node in layer:
            arr.append((node.weights, node.bias, node.layer, node.node_number))
        network.append(arr)
    # print(network)
    with open(pickle_file, "wb") as f:
        pickle.dump(network, f)

def scale_down(angles: list[float]) -> list[float]:
    return [0.25 + 0.5 * (angles[i] - gk.jointLowerLimits[i]) / (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) for i in range(len(angles))]

def scale_up(angles: list[float]) -> list[float]:
    return [gk.jointLowerLimits[i] + (angles[i] - 0.25) * (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) / 0.5 for i in range(len(angles))]

def random_pair():
    angles = [random.uniform(gk.jointLowerLimits[i], gk.jointUpperLimits[i]) for i in range(len(gk.jointLowerLimits))]
    position = list(gk.Mat2Pose(gk.forwardKinematics(np.array(angles))))
    scaled_angles = scale_down(angles)
    return position, angles, scaled_angles

def predict(nodes: list[list[Node]], inputs):
    for layer in nodes:
        out = [node.output(inputs) for node in layer]
    return out

def update_params(nodes, expected, actual, loss_derivative, inputs):
    losses = [loss_derivative(expected[i], actual[i]) for i in range(len(expected))]
    # print(expected)
    # print(actual)
    # print(losses)
    for layer in reversed(nodes):
        for node in layer:
            node.update(losses, inputs)

if __name__ == "__main__":
    # Syntax = python neural_net_multiple_pass.py <hidden layers> <width> <activation_func_number> <loss_func_number> <training size> <epochs> <load?> <save?> 

    # Set up parameters
    hidden_layers = 1
    activation_func_number = 0
    loss_func_number = 0
    set_size = 1000
    epochs = 100
    load = True
    save = True
    width = 12
    if len(sys.argv) > 1:
        hidden_layers = int(sys.argv[1])
        width = int(sys.argv[2])
        activation_func_number = int(sys.argv[3])
        loss_func_number = int(sys.argv[4])
        set_size = int(sys.argv[5])
        epochs = int(sys.argv[6])
        load = sys.argv[7].lower() == 'true'
        save = sys.argv[8].lower() == 'true'

    pickle_file = f"nnmp_h{hidden_layers}_w{width}_a{activation_func_number}_l{loss_func_number}"
    log_file = f"nnmp_log_h{hidden_layers}_w{width}_a{activation_func_number}_l{loss_func_number}.txt"

    if load: # Retrieve from file
        with open(pickle_file, "rb") as f:
            network_data = pickle.load(f)
        print(network_data)
        nodes = []
        prev_size = input_vector_size
        for layer in network_data:
            arr = []
            for node_data in layer:
                weights, bias, layer_number, node_number = node_data
                node = Node(activation_functions[activation_func_number], derivatives[activation_func_number], prev_size, nodes, layer_number, node_number)
                node.weights = weights
                node.bias = bias
                arr.append(node)
            nodes.append(arr)
            prev_size = len(arr)
    else: # Create new network
        nodes = []
        prev_size = input_vector_size
        for layer in range(hidden_layers + 1):
            arr = []
            
            if layer == hidden_layers:
                num_nodes = output_vector_size
            else:
                num_nodes = width

            for node_number in range(num_nodes):
                arr.append(Node(activation_functions[activation_func_number], derivatives[activation_func_number], prev_size, nodes, layer, node_number))
            nodes.append(arr)
            prev_size = len(arr)

    for layer in nodes:
        for node in layer:
            node.setup_gradients()

    training_rounds = 0
    total_angle_error = 0.0
    total_position_error = 0.0
    catastrophic_failures = 0
    total_time = 0.0
    total_deviation = 0.0

    # Build training set
    training_set = []
    for s in range(set_size):
        try:
            training_set.append(random_pair())
        except:
            break

    # TRAINING LOOP
    try:
        for _ in range(epochs):
            for position, expected_angles, expected_scaled_angles in training_set:
                training_rounds += 1
                # scaled are between 0 and 1
                position, expected_angles, expected_scaled_angles = random_pair()
                start = datetime.datetime.now()
                predicted_angles = predict(nodes, position)
                end = datetime.datetime.now()
                total_time += (end - start).total_seconds()

                # for layer in nodes:
                #     for node in layer:
                #         node.setup_gradients()

                unscaled_predicted_angles = scale_up(predicted_angles)
                # if r % 1000 == 0:
                #     print(unscaled_predicted_angles)
                #     print(expected_angles)
                #     print()
                #     print(predicted_angles)
                #     print(expected_scaled_angles)
                #     print()
                #     print()

                total_angle_error += sum([abs(expected_angles[i] - unscaled_predicted_angles[i]) for i in range(len(expected_angles))])
                total_deviation += sum([(expected_angles[i] - unscaled_predicted_angles[i]) ** 2 for i in range(len(expected_angles))])

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
                    average_position_error = total_position_error / training_rounds / 6
                    average_time = total_time / training_rounds
                    standard_deviation = math.sqrt(total_deviation / training_rounds / 5)
                    log(log_file, average_angle_error, standard_deviation, average_position_error, catastrophic_failures, average_time)

                    training_rounds = 0
                    total_angle_error = 0.0
                    total_position_error = 0.0   
                    catastrophic_failures = 0
                    total_time = 0.0
                    standard_deviation = 0.0
                    total_deviation = 0.0
                
    except Exception as e: 
        print("FAILURE")
        dump_network(pickle_file, nodes)
        raise e
