import numpy as np
from ann.activations import softmax


def _one_hot(y, classes):
    return np.eye(classes)[y]


def cross_entropy(logits, y_true):

    probs = softmax(logits)
    n = logits.shape[0]

    log_prob = -np.log(probs[np.arange(n), y_true] + 1e-9)

    return log_prob.mean()


def cross_entropy_grad(logits, y_true):

    probs = softmax(logits)
    n = logits.shape[0]

    grad = probs.copy()
    grad[np.arange(n), y_true] -= 1

    return grad / n


def mse(logits, y_true):

    probs = softmax(logits)
    one_hot = _one_hot(y_true, probs.shape[1])

    return np.mean((probs - one_hot) ** 2)


def mse_grad(logits, y_true):

    probs = softmax(logits)
    n, c = probs.shape
    target = _one_hot(y_true, c)

    diff = probs - target
    grad = np.zeros_like(probs)

    for k in range(c):

        jac = probs * (np.eye(c)[k] - probs[:, k:k+1])
        grad[:, k] = np.sum((2.0 / c) * diff * jac, axis=1)

    return grad / n


LOSS_FN = dict(cross_entropy=cross_entropy, mse=mse)
LOSS_GRAD = dict(cross_entropy=cross_entropy_grad, mse=mse_grad)