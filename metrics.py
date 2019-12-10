import numpy as np


def sparsity2(state):
    return np.sum(state["params"]["dense"][0][2]) / state["params"]["dense"][0][2].size