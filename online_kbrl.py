from utils import *
import numpy as np
import matplotlib.pyplot as plt
from environment import ContGrid
r_max = 1.0
cutoff = 1e-4
env = ContGrid()
n_actions = 4
s_dim = 2
gamma = .95
b = .01 #*env.max_s**2
n_episodes = 5000
if n_episodes == -1:
    n_mem = 100
else:
    n_mem = n_episodes*250
def value_iteration(Theta,R,NT,V,tot_sim=None):
    #A x AM
    Q = np.matmul(Theta,np.expand_dims(R+gamma*NT*V,-1)).squeeze()
    if tot_sim is not None:
        Q[tot_sim < cutoff] = r_max
    tied = np.max(Q,0)==Q
    two_or_more = tied.sum(0)>1
    best_act = np.squeeze(np.argmax(Q,0))
    if two_or_more.sum()>0:
        for i in np.nonzero(two_or_more)[0]:
            best_act[i] = np.random.choice(np.arange(n_actions)[tied[:,i]]) 
    return np.max(Q,0),best_act
def train(S,R,NT,SPrime):
    n_mem = S.shape[1] #A x M x Z
    raw = rbf(np.tile(SPrime.reshape([n_actions*n_mem,s_dim]),[n_actions,1,1]),S,b,normalize=False)
    #A x AM x M
    tot_sim = raw.sum(-1)
    Theta = rbf(np.tile(SPrime.reshape([n_actions*n_mem,s_dim]),[n_actions,1,1]),S,b)
    V = np.zeros([n_actions,n_mem])
    change = np.inf
    count = 0
    while change > 1:
    #for i in range(10):
        old_V = V.copy()
        V[:] = value_iteration(Theta,R,NT,V,tot_sim)[0].reshape([n_actions,-1])
        change = np.sum(np.abs(old_V-V))
        count += 1
    #print(count,change)
    return V,tot_sim
def select_action(S,R,NT,V,s):
    Theta = rbf(np.tile(s,[n_actions,1,1]),S,b)
    _,act = value_iteration(Theta,R,NT,V)
    return act
#hacky way to have dummys when uneven number of actions taken
#huge states far far from everything 
#S and SPrime different signs so 
#this doesnt kill exploration (dummys similar to each other)

big_no = 1e5
S = -np.ones([n_actions,n_mem,s_dim])*big_no
R = np.zeros([n_actions,n_mem])
NT = np.zeros([n_actions,n_mem])
SPrime = np.ones([n_actions,n_mem,s_dim])*big_no

#n episode store
term_count = 0
s = env.reset()
counts = np.zeros([n_actions],dtype=np.int)
for i in range(5):
    for a in range(n_actions):
        S[a,i,:] = env.observation_space.sample()
        SPrime[a,i,:],R[a,i],term,_ = env.get_transition(S[a,i,:],a)
        NT[a,i] = not term
counts[:] = 5
plt.ion()
i = 0
while term_count < n_episodes:
    n_mem = counts.max()
    V,tot_sim = train(S[:,:n_mem],R[:,:n_mem],NT[:,:n_mem],SPrime[:,:n_mem])
    a = select_action(S[:,:n_mem],R[:,:n_mem],NT[:,:n_mem],V,s)
    S[a,counts[a],:] = s
    SPrime[a,counts[a],:],R[a,counts[a]],term,_ = env.get_transition(s,a)
    NT[a,counts[a]] = not term
    s = SPrime[a,counts[a]].copy()
    counts[a] += 1
    if term:
        term_count += 1
        s = env.reset()
    #visualize
    i+=1
    if i % 10 == 0:
        plt.figure(0)
        plt.clf()
        plt.scatter(SPrime[:,:n_mem,0].flatten(),SPrime[:,:n_mem,1].flatten(),c=tot_sim.min(0))#V.flatten())
        plt.ylim((0,env.max_s))
        plt.xlim((0,env.max_s))
        plt.colorbar()
        plt.figure(1)
        plt.clf()
        cur_sim = tot_sim.mean(0).reshape([n_actions,-1])
        for action in range(n_actions):
            plt.subplot(2,2,action+1)
            plt.scatter(SPrime[action,:n_mem,0].flatten(),SPrime[action,:n_mem,1].flatten(),c=cur_sim[action])#V[action])
            plt.ylim((0,env.max_s))
            plt.xlim((0,env.max_s))
            plt.colorbar()
        plt.pause(.001)
n_mem = counts.max()
print(counts)
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


