import geometric_kinematics as gk
import random
import numpy as np

maxs = [float('-inf')] * 6
mins = [float('inf')] * 6

for i in range(100000):
    angles = [random.uniform(gk.jointLowerLimits[j], gk.jointUpperLimits[j]) for j in range(5)]
    position = gk.Mat2Pose(gk.forwardKinematics(np.array(angles)))
    for j in range(6):
        if position[j] > maxs[j]:
            maxs[j] = position[j]
        if position[j] < mins[j]:
            mins[j] = position[j]
print(maxs)
print(mins)