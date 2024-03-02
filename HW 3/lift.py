import numpy as np
from data_generator import fun

def lift(x):
    """Increase dimensionality of the vector x, producing a vector with all first-order products of components of x."""
    d = len(x)
    XOutProd = np.outer(x,x)
    return  np.concatenate((x,XOutProd[np.triu_indices(d)]))

def fun_prime(x):
    beta = np.zeros((len(x),1))
    beta = [2, -1.1, 0.7, 1.2, 0]
    B = np.zeros((len(x),len(x)))
    B[0,0] = 0.4
    B[4,4] = - 0.7
    B[3,1] = - 0.75
    B[1,3] = - 0.75
    c = 1.3
    # Y = beta @ x.T + x @ B @ x.T + c # Verified numerically.
    beta2 = 2 * B - np.diag(np.diag(B))
    beta2 = beta2[np.triu_indices(B.shape[0])]
    beta1 = np.concatenate((beta,beta2))
    xLift = lift(x)
    return beta1 @ xLift.T + c


if __name__ == "__main__":        
    x = np.array([0, 1, 2, 3, 4])
    print(x)
    XOutProd = np.outer(x,x)
    print(XOutProd)
    print(XOutProd[np.triu_indices(5)])
    print(lift(x))

    print('**')
    print(fun(x))
    print('**')
    print(fun_prime(x))
    