import numpy as np 

def _sigmoid(z): 
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Gather the dimensions of the X array which is the dataset values 
    N, D = X.shape
    # We take the weights which is the length of the dimension of the array : init to 0
    w = np.zeros(D)
    # init bias to 0
    b = 0.0

    for _ in range(steps):
        # we compute the weights here 
        z = X@w + b
        # compute the predicted 
        p = _sigmoid(z)
        # compute the error of predicted vs actual
        err = p - y
        # do the gradient update of the features transpose with the error to get the gradients of the weights
        grad_w = 1/N*(X.T @ err)
        # get the gradients of the bias by summing the error
        grad_b = 1/N*(np.sum(err))
        # update the weight in the direction of the learning rate with the grad w
        w = w - lr * grad_w
        # update the weight in the direction of the learning rate with the grad b
        b = b- lr * grad_b
    
    return w, b


        


