import numpy as np
import matplotlib.pyplot as plt
def rotate_box(angle, center, length, width):
    corners = np.array([[-length/2, -width/2],
                            [length/2, -width/2],
                            [length/2, width/2],
                            [-length/2, width/2]])
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])

    return np.dot(rot, corners.T)+center[:, None]
