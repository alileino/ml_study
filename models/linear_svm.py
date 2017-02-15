import numpy as np

def svm_loss(W, X, y, reg):
    '''
    Inputs have dimension D, there are C classes, and there are N examples

    :param W: numpy array of shape (D, C) containing weights
    :param X: numpy array of shape (N, D) containing training examples
    :param y: numpy array of shape (N,) containing training labels
    :param reg: regularization strength
    :return: tuple (loss, gradient) where loss is a single float, and gradient is the
     analytical gradient with respect to weights W of shape (D, C)
    '''
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    S = np.dot(X, W) # shape = (N, C)

    delta = 1.

    # transform S to margin matrix
    S = (S - np.choose(y, S.T)[...,np.newaxis])+ 1

    # Compute loss
    loss += np.sum(S[S>0]) # shape = (1,)

    loss /= num_train
    loss -= 1

    loss += 0.5 * reg * np.sum(W ** 2)

    # Compute gradient
    S = np.float64(S>0) # Turn margins into indicator matrix
    weights = (np.sum(S, axis=1)+1.)  # sum over classes, shape = (N,)

    S[np.arange(num_train), y] -= weights

    dW = np.dot(S.T, X)

    dW /= num_train
    dW = dW.T
    dW += reg*W
    return loss, dW


    # margins = S - S[range(0,num_train), y].reshape(-1,1) +1 # Alternatives
    # margins = (S - S[range(num_train),y,np.newaxis]) + 1
    # S = (S - np.choose(y, S.T)[..., np.newaxis]) + delta