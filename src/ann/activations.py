import numpy as np


def _clip(x):
    return np.clip(x, -500.0, 500.0)


def relu(x):
    return np.where(x > 0, x, 0)


def relu_grad(x):
    grad = np.zeros_like(x)
    grad[x > 0] = 1.0
    return grad


def sigmoid(x):
    x = _clip(x)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    t = np.tanh(x)
    return 1 - t * t


def softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)


ACT_FN = dict(relu=relu, sigmoid=sigmoid, tanh=tanh)
ACT_GRAD = dict(relu=relu_grad, sigmoid=sigmoid_grad, tanh=tanh_grad)