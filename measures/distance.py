import numpy as np


def l2_distance_squared(X1, X2):
    num_test = X1.shape[0]
    num_train = X2.shape[0]
    dists = np.zeros((num_test, num_train))
    # broadcast duplicating by column
    dists += np.sum(X1 * X1, axis=1).reshape(-1, 1)
    # broadcast duplicating by row
    dists += np.sum(X2 * X2, axis=1)
    dists -= 2 * np.matmul(X1, X2.transpose())

    return dists


def c_index(truey, predy):
    truey = truey.reshape(-1,1)
    predy = predy.reshape(-1,1)
    n = 0
    h_sum = 0
    for i in range(len(truey)):
        t = truey[i][0]
        p = predy[i][0]

        true_slice = truey[i+1:]
        pred_slice = predy[i+1:]

        n += np.sum(t != true_slice)

        h_sum += np.sum(((p < pred_slice) & (t < true_slice))
                        | ((p > pred_slice) & (t > true_slice)))

        h_sum += np.sum(p==pred_slice)*0.5
    if n != 0:
        return h_sum/n
    return 1


def numerical_gradient(f, x, h=1e-5):
    fx = f(x)
    grad = np.zeros(x.shape)

    it = np.nditer(x, flags="multi_index", op_flags="readwrite")
    while not it.finished:
        ix = it.multi_index
        old_value = x[ix]
        x[ix] += h
        grad[ix] = (f(x)-fx)/h
        x[ix] = old_value
        it.iternext()
    return grad