import geometric_kinematics as gk
import random
import numpy as np
import datetime

#Calculate maxes and mins of position outputs for scaling
# maxs = [float('-inf')] * 6
# mins = [float('inf')] * 6

# for i in range(100000):
#     angles = [random.uniform(gk.jointLowerLimits[j], gk.jointUpperLimits[j]) for j in range(5)]
#     position = gk.Mat2Pose(gk.forwardKinematics(np.array(angles)))
#     for j in range(6):
#         if position[j] > maxs[j]:
#             maxs[j] = position[j]
#         if position[j] < mins[j]:
#             mins[j] = position[j]
# print(maxs)
# print(mins)

# Calculate average time/error for IK
total_time = 0.0
trials = 1000
total_angle_error = 0.0
total_angle_deviation = 0.0
total_position_error = 0.0

for _ in range(trials):
    legal = False
    while not legal:
        angles = (np.random.random(5) - 0.5) * np.pi
        legal = True
        for j in range(len(angles)):
            if angles[j] < gk.jointLowerLimits[j] or angles[j] > gk.jointUpperLimits[j]:
                legal = False
    position = gk.Mat2Pose(gk.forwardKinematics(angles))
    try:
        start = datetime.datetime.now()
        result = gk.inverseKinematics(position, angles)
        end = datetime.datetime.now()
    except: # Not every combination of legal angles is actually doable--stuff can clip through itself
        print("BAD COMBINATION")
        continue
    total_time += (end - start).total_seconds()
    total_angle_error += sum(abs(angles[i] - result[i]) for i in range(len(angles)))
    total_angle_deviation += sum((angles[i] - result[i]) ** 2 for i in range(len(angles)))
    predicted_position = gk.Mat2Pose(gk.forwardKinematics(np.array(result)))
    total_position_error += sum(abs(position[i] - predicted_position[i]) for i in range(6))

print("Average Time:", total_time / trials)
print("Average Angle Error:", total_angle_error / trials / 5)
print("Angle Standard Deviation:", (total_angle_deviation / trials / 5))
print("Average Position Error:", total_position_error / trials / 6)
