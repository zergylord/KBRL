import numpy as np
class ActionSpace(object):
    def __init__(self,n):
        self.n = n
        self.sample = lambda: np.random.randint(n)

class ObservationSpace(object):
    def __init__(self,shape,sample = None):
        self.shape = shape
        if sample is None:
            self.sample = lambda: np.random.randn(shape)
        else:
            self.sample = sample

class Environment(object):
    toggle_screen = lambda x:None

    def prep(self,samples):
        return samples

    def __init__(self, n_actions):
        self.action_labels = [str(i) for i in range(n_actions)]
        self.action_set = range(n_actions)
        self.replay_dtype = np.float32

    def plot_state(self, ax, state, prev_ret):
        # TODO: implement
        return None
class ContGrid(Environment):
    max_s = 1
    observation_space = ObservationSpace(2,lambda: np.random.rand(2)*ContGrid.max_s)
    def __init__(self,n_actions = 4):
        super().__init__(n_actions)
        self.goal = np.array([.1,.5])*self.max_s
        print(self.goal)
        self.goal_size = .15*self.max_s
        self.reset()
        self.rad_inc = 2*np.pi/n_actions
        self.action_space = ActionSpace(n_actions)
        self.radius = .1*self.max_s
    def reset(self):
        #self.s = np.asarray([.1,.1])
        self.s = np.random.rand(2)*self.max_s
        return self.s
    def get_transition(self,s,a):
        term = False
        r = 0
        sPrime =  s \
            + np.asarray([self.radius*np.cos(self.rad_inc*a), \
                self.radius*np.sin(self.rad_inc*a)]) \
            +np.random.randn(2)*self.radius/10
        if np.any(sPrime > self.max_s) or np.any(sPrime < 0):
            sPrime[sPrime >= self.max_s] = self.max_s-.01*self.max_s
            sPrime[sPrime < 0] = 0+.01*self.max_s
            #r = -1.0
        elif np.all(sPrime > self.goal) and np.all(sPrime < (self.goal+self.goal_size)):
            pass
            #r = 1
            #term = True
        return sPrime.copy(),r,term,False
    def step(self,a):
        self.s,r,term,_ = self.get_transition(self.s,a)
        return sPrime.copy(),r,term,False
