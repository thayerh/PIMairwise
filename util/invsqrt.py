import numpy as np


def fast_inv_sqrt(x):
    """
    Fast approximation of 1/sqrt(x) using the exponent trick

    :param x: float or numpy array of floats
    :return: approximate 1/sqrt(x)
    """
    x = np.asarray(x, dtype=np.float32)
    if np.any(x <= 0):
        raise ValueError("Input must be positive")

    xi = x.view(np.int32)
    magic = 0x5f3759df
    y = magic - (xi >> 1)
    y = y.view(np.float32)

    return y

if __name__ == "__main__":
    print(1/fast_inv_sqrt(2))