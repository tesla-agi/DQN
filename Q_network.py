import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self,obs_dim,n_actions,hidden_dim=128,dueling=False):
        super().__init__()
        self.dueling=dueling
        self.trunk=nn.Sequential(
            nn.Linear(obs_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(),
        )

        if dueling:
            self.value_head=nn.Linear(hidden_dim,1)
            self.advantage_head=nn.Linear(hidden_dim,n_actions)

        else:
            self.q_head=nn.Linear(hidden_dim,n_actions)


    def forward(self,x):
        features=self.trunk(x)
        if self.dueling:
            v=self.value_head(features)
            a=self.advantage_head(features)
            q=v+(a-a.mean(dim=1,keepdim=True))

        else:
            q=self.q_head(features)


        return q






