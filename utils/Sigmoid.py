import math
import numpy as np


def sigmoid(theta, x):
    sum = 0.0
    for i in range(len(x)):
        sum -= theta[i] * x[i]
    return 1 / (1 + math.exp(sum))


def sigmoid2(theta, x):
    return 1 / (1 + np.exp(np.dot(theta, x)))

