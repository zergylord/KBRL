import numpy as np
def softmax(X):
    X = np.exp(X - X.max(-1,keepdims=True))
    return X/X.sum(-1,keepdims=True)
#cdist for matrices with arbitary num of batch ranks
def cdist(X,Y):
    rank = len(X.shape)
    XX = np.sum(np.square(X),-1,keepdims=True)
    YY = np.expand_dims(np.sum(np.square(Y),-1),-2)
    if rank == 2:
        XY = -2*np.matmul(X,np.transpose(Y))
    elif rank == 3:
        XY = -2*np.matmul(X,np.transpose(Y,[0,2,1]))
    elif rank == 3:
        XY = -2*np.matmul(X,np.transpose(Y,[0,1,3,2]))
    else:
        print("FOOBAR")
    return XX+YY+XY
def rbf(X,Y,b=1.0,normalize=True):
    raw = -cdist(X,Y)/b
    if normalize:
        return softmax(raw)
    else:
        return np.exp(raw)
        #return raw
