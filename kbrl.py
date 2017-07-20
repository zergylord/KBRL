from utils import *
import numpy as np
import matplotlib.pyplot as plt
from environment import ContGrid
r_max = 1.0
def value_iteration(Theta,R,NT,V,tot_sim=None):
    Q = np.matmul(Theta,np.expand_dims(R+gamma*NT*V,-1))
    if tot_sim is not None:
        Q[tot_sim < 1e-3] = r_max
    return np.max(Q,0).reshape([n_actions,-1]),np.squeeze(np.argmax(Q,0))

env = ContGrid()
n_actions = 4
s_dim = 2
gamma = .95
b = .001 #*env.max_s**2
n_episodes = 5
if n_episodes == -1:
    n_mem = 100
else:
    n_mem = n_episodes*250
S = np.random.randn(n_actions,n_mem,s_dim)
R = np.random.randn(n_actions,n_mem)
NT = np.zeros([n_actions,n_mem])
SPrime = np.random.randn(n_actions,n_mem,s_dim)
if n_episodes == -1:
    #uniform sampling
    for i in range(n_mem):
        for a in range(n_actions):
            S[a,i,:] = env.observation_space.sample()
            SPrime[a,i,:],R[a,i],term,_ = env.get_transition(S[a,i,:],a)
            NT[a,i] = not term
    #NT = np.ones([n_actions,n_mem])
else:
    #n episode store
    term_count = 0
    s = env.reset()
    counts = np.zeros([n_actions],dtype=np.int)
    while term_count < n_episodes:
        a = env.action_space.sample()
        S[a,counts[a],:] = s
        SPrime[a,counts[a],:],R[a,counts[a]],term,_ = env.get_transition(s,a)
        NT[a,counts[a]] = not term
        s = SPrime[a,counts[a]].copy()
        counts[a] += 1
        if term:
            term_count += 1
            s = env.reset()
    n_mem = counts.max()
    print(counts)
    for a in range(n_actions):
        while counts[a] < n_mem:
            S[a,counts[a],:] = env.observation_space.sample()
            SPrime[a,counts[a],:],R[a,counts[a]],term,_ = env.get_transition(S[a,counts[a],:],a)
            NT[a,counts[a]] = not term
            counts[a] += 1
    print(counts)
    S = S[:,:n_mem]
    R = R[:,:n_mem]
    SPrime = SPrime[:,:n_mem]
    print(SPrime.shape)
    NT = NT[:,:n_mem]




#"train"
raw = rbf(np.tile(SPrime.reshape([n_actions*n_mem,s_dim]),[n_actions,1,1]),S,b,normalize=False)
tot_sim = raw.sum(-1)
print(tot_sim.shape,'ager')
print(np.sum(tot_sim < .1))
print(tot_sim.min(),tot_sim.mean(),tot_sim.max())
Theta = rbf(np.tile(SPrime.reshape([n_actions*n_mem,s_dim]),[n_actions,1,1]),S,b)
print('Theta:',Theta.shape)
V = np.zeros([n_actions,n_mem])
change = np.inf
count = 0
while change > 1:
    old_V = V.copy()
    V[:],_ = value_iteration(Theta,R,NT,V,tot_sim)
    change = np.sum(np.abs(old_V-V))
    count += 1
    print(count,change)

#visualize
plt.subplot(2,2,1)
plt.scatter(SPrime[:,:,0].flatten(),SPrime[:,:,1].flatten(),c=np.float32(tot_sim.min(0)>1e-3))#V.flatten())
plt.ylim((0,env.max_s))
plt.xlim((0,env.max_s))
plt.colorbar()
#plt.show()
'''
for a in range(n_actions):
    plt.scatter(SPrime[a,:,0],SPrime[a,:,1],c=V[a])
    print(V[a].min(),V[a].max())
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.show()
'''

grid_side = 40
X,Y = np.meshgrid(np.arange(0,env.max_s,env.max_s/grid_side),
        np.arange(0,env.max_s,env.max_s/grid_side))
print('hi')
S_test = np.asarray([X.flatten(),Y.flatten()]).transpose()
Theta_test = rbf(np.tile(S_test,[n_actions,1,1]),S,b)
print(Theta_test.shape)
V_test,policy = value_iteration(Theta_test,R,NT,V)
plt.subplot(2,2,2)
plt.scatter(S_test[:,0],S_test[:,1],c=V_test.flatten())
plt.ylim((0,env.max_s))
plt.xlim((0,env.max_s))
print('hi')
plt.subplot(2,2,3)
plt.hist(policy)
plt.subplot(2,2,4)
x_dir = np.zeros([grid_side**2])
y_dir = np.zeros([grid_side**2])
for i in range(grid_side**2):
    #right
    if policy[i] == 0:
        x_dir[i] = 1
    #up
    elif policy[i] == 1:
        y_dir[i] = 1
    #left
    elif policy[i] == 2:
        x_dir[i] = -1
    #down
    elif policy[i] == 3:
        y_dir[i] = -1
plt.quiver(X,Y,x_dir.reshape([grid_side,grid_side]), \
        y_dir.reshape([grid_side,grid_side]))
plt.show()


