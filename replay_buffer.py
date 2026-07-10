import numpy as np

class ReplayBuffer:
    def __init__(self,capacity=100000,obs_dim=8,correlated=False):
        self.obs_dim=obs_dim
        self.capacity=capacity
        self.correlated=correlated
        self.ptr=0                                      #Push
        self.size=0                                     #Sample

        self.states=np.zeros((capacity,obs_dim),dtype=np.float32)
        self.actions=np.zeros(capacity,dtype=np.int64)
        self.rewards=np.zeros(capacity,dtype=np.float32)
        self.next_states=np.zeros((capacity,obs_dim),dtype=np.float32)
        self.dones=np.zeros(capacity,dtype=np.float32)


    def push(self,state,action,reward,next_state,done):
        idx=self.ptr
        self.states[idx]=state
        self.actions[idx]=action
        self.rewards[idx]=reward
        self.next_states[idx]=next_state
        self.dones[idx]=done

        self.ptr=(idx+1)%self.capacity
        self.size=min(self.size+1,self.capacity)


    def sample(self,batch_size):
        if self.correlated:
            indices=np.arange((self.ptr-batch_size),self.ptr)%self.capacity    #self.capacity used because it won't give -ve indices
        else:
            indices=np.random.randint(0,self.size,size=batch_size)

        states=self.states[indices]
        actions=self.actions[indices]
        rewards=self.rewards[indices]
        next_states=self.next_states[indices]
        dones=self.dones[indices]

        return states,actions,rewards,next_states,dones

    def __len__(self):
        return self.size
