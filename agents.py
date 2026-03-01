from torch import nn
import torch

class MLPAgent(nn.Module):
    def __init__(self, in_dim=None, out_dim=None, discr=False, hidden_sizes=(128, 128), init_log_std=0.2, learning_rate=3e-4):
        super().__init__()
        
        self.in_dim = in_dim
        last_size = hidden_sizes[-1]
        self.shared_net = self._build_trunk(in_dim, hidden_sizes)
        self.value_net = self._build_trunk(in_dim, hidden_sizes)

        self.discr = discr
        
        if discr:
            self.out_head = nn.Linear(last_size, out_dim)
            nn.init.orthogonal_(self.out_head.weight, gain=0.01)
        else:
            self.mu_head = nn.Linear(last_size, out_dim)
            nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
            nn.init.constant_(self.mu_head.bias, 0.0)
            self.log_std = nn.Parameter(torch.ones(out_dim) * init_log_std)

        self.value_head = nn.Linear(last_size, 1)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

        pi_params = list(self.shared_net.parameters())
        if self.discr:
            pi_params += list(self.out_head.parameters())
        else:
            pi_params += list(self.mu_head.parameters()) + [self.log_std]
        
        learning_rate = float(learning_rate)
        self.optimizer = torch.optim.Adam([
                {'params': pi_params, 'lr': learning_rate},
                {'params': self.value_net.parameters(), 'lr': learning_rate * 3},
                {'params': self.value_head.parameters(), 'lr': learning_rate * 3},
            ],
            lr=float(learning_rate))
    
    def _build_trunk(self, in_dim, hidden_sizes):
        layers = []
        last_size = in_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.Tanh())
            last_size = size
        net = nn.Sequential(*layers)
        for layer in net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.constant_(layer.bias, 0.0)
        return net
    
    def forward(self, obs):
        x = self.shared_net(obs)
        if self.discr:
            x = self.out_head(x)
            return nn.functional.softmax(x, dim=-1)
        else:
            mu = self.mu_head(x)
            std = torch.exp(self.log_std)
            std = torch.clamp(std, min=1e-3, max=1.0)
            return mu, std
    
    def value(self, obs):
        x = self.value_net(obs)
        return self.value_head(x)
