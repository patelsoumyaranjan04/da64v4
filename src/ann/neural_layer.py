import numpy as np
from ann.activations import ACT_FN, ACT_GRAD


class Layer:

    def __init__(self, in_dim, out_dim, activation, weight_init):

        self.activation = activation
        self.W, self.b = self._init_weights(in_dim, out_dim, weight_init)

        self.grad_W = None
        self.grad_b = None

        self.cache_input = None
        self.cache_z = None

    def _init_weights(self, in_dim, out_dim, method):

        if method == "xavier":
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            W = np.random.uniform(-limit, limit, (in_dim, out_dim))
        else:
            W = np.random.randn(in_dim, out_dim) * 0.01

        b = np.zeros((1, out_dim))

        return W, b

    def forward(self, x):

        self.cache_input = x
        z = np.dot(x, self.W) + self.b
        self.cache_z = z

        if self.activation is None:
            return z

        return ACT_FN[self.activation](z)

    def backward(self, delta):

        if self.activation is not None:
            delta = delta * ACT_GRAD[self.activation](self.cache_z)

        self.grad_W = self.cache_input.T @ delta
        self.grad_b = delta.sum(axis=0, keepdims=True)

        return delta @ self.W.T