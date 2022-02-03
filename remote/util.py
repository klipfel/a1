import numpy as np


def adapt_data_for_comm(array):
    """
    Adapts data for comm, takes a row numpy array.
    :return:
    """
    return array.flatten().tolist()


def recover_data(data):
    """Puts data back in row array."""
    return np.reshape(np.array(data, dtype=np.float32), (1, -1))
