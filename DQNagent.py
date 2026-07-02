import random

import torch
from Q_network import QNetwork
from replay_buffer import ReplayBuffer

class DQNAgent:
    def __init__(self,obs_dim,n_actions,hidden_dim=(128,128),dueling=False,use_double_dqn=False,gamma=0.99,lr=3e-4,
                 batch_size=64,grad_clip=10.0,target_upd_freq=1000,eps_start=1.0,eps_end=0.05,eps_decay_steps=50000,
                 buffer_capacity=100000,device='mps'):

        self.gamma=gamma
        self.lr=lr
        self.batch_size=batch_size
        self.target_upd_freq=target_upd_freq
        self.grad_clip=grad_clip
        self.n_actions=n_actions
        self.device=device

        self.q_online=QNetwork(obs_dim,n_actions,hidden_dim,dueling=dueling).to(device)
        self.q_target=QNetwork(obs_dim,n_actions,hidden_dim,dueling=dueling).to(device)
        self.q_target.load_state_dict(self.q_online.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad=False
        self.q_target.eval()
        self.optimizer=torch.optim.Adam(self.q_online.parameters(),lr=lr)
        self.replay_buffer=ReplayBuffer(capacity=buffer_capacity,obs_dim=obs_dim)
        self.step_count=0

        self.use_double_dqn=use_double_dqn
        self.eps_start=eps_start
        self.eps_end=eps_end
        self.eps_decay_steps=eps_decay_steps

    def select_action(self,state,greedy=False):
        state=torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        eps=max(self.eps_end,self.eps_start-(self.eps_start-self.eps_end)*(self.step_count/self.eps_decay_steps))
        if greedy:
            with torch.no_grad():
                action=self.q_online(state).argmax(dim=1).item()
            return action

        self.step_count+=1

        if random.random()<eps:
            return random.randrange(self.n_actions)
        else:
            with torch.no_grad():
                return self.q_online(state).argmax(dim=1).item()








