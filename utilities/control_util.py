import numpy as np


def error(x, y):
    np.linalg.norm(np.array(x-y))
