import numpy as np

def softmax(x):
    x = np.array(x)
    # Subtract max for numerical stability, keepdims for broadcasting over 1D and 2D
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)