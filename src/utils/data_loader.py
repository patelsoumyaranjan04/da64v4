import numpy as np
from sklearn.model_selection import train_test_split
from keras.datasets import mnist, fashion_mnist


def _prepare(X):

    X = X.reshape(len(X), -1)
    X = X.astype(np.float32) / 255.0

    return X


def load_data(name, val_split=0.1, seed=42):

    if name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = _prepare(X_train)
    X_test = _prepare(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=val_split,
        random_state=seed,
        stratify=y_train
    )

    return X_train, y_train, X_val, y_val, X_test, y_test