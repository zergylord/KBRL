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
    observation_space = ObservationSpace(2,lambda: np.random.rand(2))
    def __init__(self,n_actions = 4):
        super().__init__(n_actions)
        self.goal = np.array([.5,.5])
        self.goal_size = .15
        self.reset()
        self.rad_inc = 2*np.pi/n_actions
        self.action_space = ActionSpace(n_actions)
        self.radius = .1
    def reset(self):
        #self.s = np.asarray([.1,.1])
        self.s = np.random.rand(2)
        return self.s
    def get_transition(self,s,a):
        term = False
        r = 0
        sPrime =  s + np.asarray([self.radius*np.cos(self.rad_inc*a),self.radius*np.sin(self.rad_inc*a)]) +np.random.randn(2)*self.radius/10
        if np.any(sPrime > 1) or np.any(sPrime < 0):
            sPrime[sPrime >= 1] = 1-.01
            sPrime[sPrime < 0] = 0+.01
            r = -1.0
            #term = True
        elif np.all(sPrime > self.goal) and np.all(sPrime < (self.goal+self.goal_size)):
            r = 1
            term = True
        return sPrime.copy(),r,term,False
    def step(self,a):
        self.s,r,term,_ = self.get_transition(self.s,a)
        return sPrime.copy(),r,term,False
