import sys
import pickle
import random
import geometric_kinematics as gk
import numpy as np
import datetime
import math

position_approx_maxs = [1, 1, 1, 4, 2, 4]
position_approx_mins = [-0.5, -1, -0.5, -4, -2, -4]

def log(log_file, average_angle_error, standard_deviation, average_position_error, catastrophic_failures, average_time):
    with open(log_file, "a") as f:
        f.write(f"{average_angle_error},{standard_deviation},{average_position_error},{catastrophic_failures},{average_time}\n")
    print(f"Avg Angle Error: {average_angle_error}, Standard Deviation: {standard_deviation}, Avg Position Error: {average_position_error}, Catastrophic Failures: {catastrophic_failures}")#, Avg Time: {average_time}")

def is_catastrophic_failure(angles: list[float]) -> bool:
    for i in range(len(angles)):
        if angles[i] < gk.jointLowerLimits[i] or angles[i] > gk.jointUpperLimits[i]:
            # print(i)
            # print(angles[i])
            # print()
            return True
    return False

def dump_polys(pickle_file, polynomials, constants):
    data = (polynomials, constants)
    with open(pickle_file, "wb") as f:
        pickle.dump(data, f)

def scale_down_angles(angles: list[float]) -> list[float]:
    return [(angles[i] - gk.jointLowerLimits[i]) / (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) for i in range(len(angles))]

def scale_up_angles(angles: list[float]) -> list[float]:
    return [gk.jointLowerLimits[i] + angles[i] * (gk.jointUpperLimits[i] - gk.jointLowerLimits[i]) for i in range(len(angles))]

def scale_down_position(position: list[float]) -> list[float]:
    return [(position[i] - position_approx_mins[i]) / (position_approx_maxs[i] - position_approx_mins[i]) for i in range(len(position))]

def scale_up_position(position: list[float]) -> list[float]:
    return [position_approx_mins[i] + position[i] * (position_approx_maxs[i] - position_approx_mins[i]) for i in range(len(position))]

if __name__ == "__main__":
    # Syntax = python polynomial_regression.py <degree> <learning rate> <training rounds> <load?> <save?> 

    # Set up parameters
    degree = 3
    learning_rate = 0.01
    rounds = 1000
    load = True
    save = True
    if len(sys.argv) > 1:
        degree = int(sys.argv[1])
        learning_rate = float(sys.argv[2])
        rounds = int(sys.argv[3])
        load = sys.argv[4].lower() == 'true'
        save = sys.argv[5].lower() == 'true'

    pickle_file = f"poly_deg{degree}_lr{learning_rate}"
    log_file = f"poly_log_deg{degree}_lr{learning_rate}.txt"

    # Generate all variable combinations up to degree as sets of original indices
    temp_combinations = [{tuple([0]), tuple([1]), tuple([2]),tuple([3]),tuple([4]),tuple([5]),}] # Start with degree 1
    for d in range(1, degree):
        new_degree = set()
        for var in range(6):
            for combo in temp_combinations[-1]:
                new_combo = list(combo)
                new_combo.append(var)
                new_combo.sort()
                new_combo = tuple(new_combo)
                new_degree.add(new_combo) # prevents dupes
        temp_combinations.append(new_degree)
    combinations = []
    for s in temp_combinations:
        combinations.extend(s)
        
    if load: # Retrieve from file
        with open(pickle_file, "rb") as f:
            data = pickle.load(f)
        print(data)
        coefficients, constants = data
    else: # Create new model
        coefficients = [[0 for _ in range(len(combinations))] for i in range(5)] # 5 outputs need 5 polynomials
        constants = [0 for _ in range(5)]

    training_rounds = 0
    total_angle_error = 0.0
    total_position_error = 0.0
    catastrophic_failures = 0
    total_time = 0.0
    total_deviation = 0.0

    try:
        # Training and evaluation code would go here
        for r in range(rounds):
            training_rounds += 1

            # Create in-out pair
            angles = [random.uniform(gk.jointLowerLimits[i], gk.jointUpperLimits[i]) for i in range(len(gk.jointLowerLimits))]
            position = list(gk.Mat2Pose(gk.forwardKinematics(np.array(angles))))
            scaled_angles = scale_down_angles(angles)
            scaled_position = scale_down_position(position)

            terms = []
            for combo in combinations:
                term = 1.0
                for index in combo:
                    term *= scaled_position[index]
                terms.append(term)
            
            start = datetime.datetime.now()
            outputs = [sum([coefficients[i][j] * terms[j] for j in range(len(terms))]) + constants[i] for i in range(5)]
            scaled_up_outputs = scale_up_angles(outputs)
            end = datetime.datetime.now()
            total_time += (end - start).total_seconds()
            
            total_angle_error += sum(abs(angles[i] - scaled_up_outputs[i]) for i in range(len(angles)))
            total_deviation += sum((angles[i] - scaled_up_outputs[i]) ** 2 for i in range(len(angles)))
            try:
                predicted_position = gk.Mat2Pose(gk.forwardKinematics(np.array(scaled_up_outputs)))
                total_position_error = sum([abs(position[i] - predicted_position[i]) for i in range(6)])
            except:
                total_position_error += total_position_error / training_rounds
            
            if is_catastrophic_failure(outputs):
                    catastrophic_failures += 1
            # Update coefficients and constants
            for i in range(5):
                err = scaled_angles[i] - outputs[i]
                for j in range(len(terms)):
                    coefficients[i][j] += learning_rate * err * terms[j]
                constants[i] += learning_rate * err

            if training_rounds % 100 == 0:
                # Log results and save network
                if save:
                    dump_polys(pickle_file, coefficients, constants)

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
        dump_polys(pickle_file, coefficients, constants)
        raise e
        
        