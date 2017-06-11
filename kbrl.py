import numpy as np
import matplotlib.pyplot as plt
from environment import ContGrid
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
def value_iteration(Theta,R,NT,V):
    Q = np.matmul(Theta,np.expand_dims(R+gamma*NT*V,-1))
    return np.max(Q,0).reshape([n_actions,n_mem])

n_actions = 4
n_mem = 1000
s_dim = 2
gamma = .95
b = .001
S = np.random.randn(n_actions,n_mem,s_dim)
R = np.random.randn(n_actions,n_mem)
NT = np.zeros([n_actions,n_mem])
SPrime = np.random.randn(n_actions,n_mem,s_dim)
env = ContGrid()
for i in range(n_mem):
    for a in range(n_actions):
        S[a,i,:] = env.observation_space.sample()
        SPrime[a,i,:],R[a,i],term,_ = env.get_transition(S[a,i,:],a)
        NT[a,i] = not term
#NT = np.ones([n_actions,n_mem])

log_rbf = -cdist( \
        np.tile(SPrime.reshape([n_actions*n_mem,s_dim]),[n_actions,1,1]) \
        ,S)/b
Theta = np.exp(log_rbf)/np.exp(log_rbf).sum(-1,keepdims=True)
V = np.zeros([n_actions,n_mem])
change = np.inf
count = 0
while change > 1:
    old_V = V.copy()
    V[:] = value_iteration(Theta,R,NT,V)
    change = np.sum(np.abs(old_V-V))
    count += 1
    print(count,change)
for a in range(n_actions):
    plt.scatter(SPrime[a,:,0],SPrime[a,:,1],c=V[a])
    print(V[a].min(),V[a].max())
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.show()


