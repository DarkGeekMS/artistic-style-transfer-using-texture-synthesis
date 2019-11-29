import numpy as np

def IRLS(R, X, Z, I_irls, r):
    """ 
    The function to perform IRLS for least square optimization
    between style and stylized image patches. 

    Parameters: 
    R (ndarray): array of patches from the stylized image X ----> X
    X (ndarray): the initial estimate of stylized image ----> Beta
    Z (ndarray): array of style patches (of the same resolution as current X) ----> Y 
    I_irls (int): number of IRLS iterations
    r (int): the robust statistic value

    Returns: 
    ndarray: the temporal estimate Xtilde

    """   
    cur_X = X # current estimate
    w = 1 # initial weight
    W = diag(repeat(w, R.shape[0])) # initial weight matrix
    cur_X = dot(inv(R.T.dot(W).dot(R)), ( R.T.dot(W).dot(Z))) # initial estimate update
    for i in range(I_irls): # loop over IRLS iterations
        w_prev = w 
        w = 1 / max(0.0001, w_prev) # compute new weight
        W = diag(repeat(w, R.shape[0])) # form new weight matrix
        cur_X = dot(inv(X.T.dot(W).dot(X)), ( X.T.dot(W).dot(Z))) # estimate update

    Xtilde = cur_X # new Xtilde 
    return Xtilde   