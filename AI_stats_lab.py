"""
AI_stats_lab.py

Neural Networks Lab: 3-Layer Forward Pass and Backpropagation

Implement all functions.
Do NOT change function names.
Do NOT print inside functions.
"""

import numpy as np


def sigmoid(z):
    """
    sigmoid(z) = 1 / (1 + exp(-z))
    """
    pass


def forward_pass(X, W1, W2, W3):
    """
    3-layer neural network forward pass.

    Layer 1:
        h1 = sigmoid(XW1)

    Layer 2:
        h2 = sigmoid(h1W2)

    Output layer:
        y = sigmoid(h2W3)

    Returns:
        h1, h2, y
    """
    pass


def backward_pass(X, h1, h2, y, label, W1, W2, W3):
    """
    Backpropagation for a 3-layer sigmoid neural network.

    Returns:
        dW1, dW2, dW3, loss
    """
    pass
