def vl_myfc(X, W, dzdy=None):
    if dzdy is None:
        return W.T.dot(X_t)
    else:
        Y = W.dot(dzdy)
        Y_w = X.dot(dzdy.T)
        return Y, Y_w