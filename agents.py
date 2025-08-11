from torch import nn
import torch

class MLPAgent(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, discr=False, hidden_sizes=(128, 128), init_log_std=0.2, learning_rate=3e-4):
        super().__init__()
        
        self.in_dim = in_dim
        layers = []
        last_size = self.in_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self.shared_net = nn.Sequential(*layers)

        self.discr = discr
        
        if discr:
            self.out_head = nn.Linear(last_size, out_dim)
        else:
            self.mu_head = nn.Linear(last_size, out_dim)
            self.log_std = nn.Parameter(torch.ones(out_dim) * init_log_std)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
    
    def forward(self, obs):
        x = self.shared_net(obs)
        if self.discr:
            x = self.out_head(x)
            return nn.functional.softmax(x, dim=-1)
        else:
            mu = self.mu_head(x)
            std = torch.exp(self.log_std)
            return mu, std
        
