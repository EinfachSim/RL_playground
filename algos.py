import numpy as np
import torch
from agents import *


class REINFORCE:
    def __init__(self, model_params, eval_mode=False, gamma=0.99):
        self.discrete = model_params["discr"]
        self.pol = MLPAgent(**model_params)
        self.eval = eval_mode
        self.gamma = gamma

    def update(self, batch):

        returns = []
        log_probs = []
        for trajectory in batch:
            G_t = 0
            for r, s, a, log_prob in reversed(trajectory):
                G_t = r + self.gamma * G_t
                returns.insert(0, G_t)
                log_probs.insert(0, log_prob)
        
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        log_probs = torch.stack(log_probs)
        
        loss = -torch.sum(log_probs * returns)

        self.pol.optimizer.zero_grad()
        loss.backward()
        self.pol.optimizer.step()

    def sample_action(self, obs):
        out = self.pol(torch.Tensor(obs))

        if self.discrete:
            if self.eval:
                action = torch.argmax(out, dim=-1)
                return action.cpu().numpy()
            dist = torch.distributions.Categorical(out)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
            return (action.cpu().numpy(), log_prob)

        else:
            mu = out[0]
            std = out[1]
            if self.eval:
                return mu.detach().numpy()
            else:
                dist = torch.distributions.Normal(mu, std)
                action = dist.sample()                # stochastic during training
                log_prob = dist.log_prob(action).sum(axis=-1)  # sum over action dims
                return action.cpu().numpy(), log_prob.cpu()

        

