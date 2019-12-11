import numpy as np


def confusionMatrix(state, subset, cutoff, categories):
    """All matrices should have positive results in the first row or column;
    All matrices already have predictions occupying rows and actual occupying columns
    """

    matrix = state["eval"][subset]["hard_cutoff" if cutoff else "sigmoidal"]["confusion_matrix"]
    if categories == "any":
        matrix[[0, 1]] = matrix[[1, 0]]
        
    return matrix


def sparsity(state):
    return np.array([1 - params[2].sum() / params[2].size for params in state["params"]["dense"] if len(params) == 3])

def zeroRows(state):
    col_sums = state["params"]["dense"][0][2].sum(axis=1)
    return sum([col_sum == 0 for col_sum in col_sums])

def steepness(state):
    return state["params"]["activation"][0]

def accuracy(state, subset, cutoff):
    cm = confusionMatrix(state, subset, cutoff, categories)
    return cm.trace() / cm.sum()

def sensitivity(state, subset, cutoff, categories):
    """True positive / Condition positive"""
    cm = confusionMatrix(state, subset, cutoff, categories)
    return cm[0][0] / cm[:, 0].sum()

def specificity(state, subset, cutoff, categories):
    """True negative / Condition negative"""
    cm = confusionMatrix(state, subset, cutoff, categories)
    return cm[1][1] / cm[:, 1].sum()

def likelihoodRatio(state, subset, cutoff, categories):
    """Sensitivity / (1 - Specificity)"""
    _sensitivity = sensitivity(state, subset, cutoff, categories)
    _specificity = specificity(state, subset, cutoff, categories)
    return _sensitivity / (1 - _specificity)

def diagnosticOddsRatio(state, subset, cutoff, categories):
    """Sensitivity * Specificity / (1 - Sensitivity) / (1 - Specificity)"""
    _sensitivity = sensitivity(state, subset, cutoff, categories)
    _specificity = specificity(state, subset, cutoff, categories)
    return _sensitivity * _specificity / (1 - _sensitivity) / (1 - _specificity)