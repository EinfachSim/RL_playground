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
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        log_probs = torch.stack(log_probs)
        
        loss = -torch.mean(log_probs * returns)

        self.pol.optimizer.zero_grad()
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.pol.parameters(), max_norm=0.5)
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
            return (action.detach().numpy(), log_prob)

        else:
            mu = out[0]
            std = out[1]
            if self.eval:
                return mu.detach().numpy()
            else:
                dist = torch.distributions.Normal(mu, std)
                action = dist.rsample()                # stochastic during training
                log_prob = dist.log_prob(action).sum(axis=-1)  # sum over action dims

                return action.detach().numpy(), log_prob.cpu()

        
class PPO:
    def __init__(
        self,
        model_params,
        eval_mode=False,
        gamma=0.99,
        lam=0.95,
        clip_ratio=0.15,
        train_iters=40,
        target_kl=0.005,
        vf_coef=1.0,
        ent_coef=0.0,
    ):
        self.discrete    = model_params["discr"]
        self.pol         = MLPAgent(**model_params)   # same as REINFORCE
        self.eval        = eval_mode
        self.gamma       = gamma
        self.lam         = lam
        self.clip_ratio  = clip_ratio
        self.train_iters = train_iters
        self.target_kl   = target_kl
        self.vf_coef     = vf_coef
        self.ent_coef    = ent_coef

    def sample_action(self, obs):
        """
        Eval mode  → returns action only (no grad), identical to REINFORCE.
        Train mode → returns (action, log_prob).
        """
        out = self.pol(torch.as_tensor(obs, dtype=torch.float32))

        if self.discrete:
            if self.eval:
                return torch.argmax(out, dim=-1).cpu().numpy()
            dist     = torch.distributions.Categorical(out)
            action   = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1).detach()
            return action.detach().numpy(), log_prob

        else:  # continuous
            mu, std = out[0], out[1]
            if self.eval:
                return mu.detach().numpy()
            dist     = torch.distributions.Normal(mu, std)
            action   = dist.rsample()
            log_prob = dist.log_prob(action).sum(axis=-1).detach()
            return action.detach().numpy(), log_prob.cpu()

    def update(self, batch):
        """
        Drop-in replacement for REINFORCE.update.

        batch : list of trajectories
                each trajectory : list of (r, s, a, log_prob) tuples
                  r        – scalar reward
                  s        – observation at that step
                  a        – action taken (numpy or tensor)
                  log_prob – log π_old(a|s), detached tensor (from sample_action)
        """
        # ── 1. Unpack batch and compute GAE-λ advantages ──────────────────
        obs_list, act_list, adv_list, ret_list, logp_old_list = [], [], [], [], []

        for trajectory in batch:
            obs_traj, act_traj, rew_traj, logp_traj = self._unpack_trajectory(trajectory)
            advantages, returns = self._gae(obs_traj, rew_traj, last_val=0.0)

            obs_list.extend(obs_traj)
            act_list.extend(act_traj)
            adv_list.extend(advantages)
            ret_list.extend(returns)
            logp_old_list.extend(logp_traj)

        obs_t      = torch.as_tensor(obs_list, dtype=torch.float32)
        act_t      = self._as_action_tensor(act_list)
        adv_t      = torch.tensor(adv_list,    dtype=torch.float32)
        ret_t      = torch.tensor(ret_list,    dtype=torch.float32)
        logp_old_t = torch.stack(logp_old_list).detach()

        # Normalise advantages (same spirit as REINFORCE's return normalisation)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        # ── 2. PPO gradient steps with early stopping ─────────────────────
        for i in range(self.train_iters):
            logp_new, v_new, entropy = self._evaluate(obs_t, act_t)

            # Clipped surrogate objective
            ratio   = torch.exp(logp_new - logp_old_t)
            clipped = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
            loss_pi = -torch.min(ratio * adv_t, clipped * adv_t).mean()

            # Value function loss
            loss_v = ((v_new - ret_t) ** 2).mean()
            loss = loss_pi + self.vf_coef * loss_v - self.ent_coef * entropy

            self.pol.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pol.parameters(), max_norm=0.5)
            self.pol.optimizer.step()

            # Early stopping: if policy moved too far, stop updating
            with torch.no_grad():
                approx_kl = (logp_old_t - logp_new).mean().item()
            if approx_kl > 1.5 * self.target_kl:
                break


    # ── Private helpers ───────────────────────────────────────────────────────

    def _unpack_trajectory(self, trajectory):
        """Split list of (r, s, a, log_prob) into four separate lists."""
        obs_list, act_list, rew_list, logp_list = [], [], [], []
        for r, s, a, log_prob in trajectory:
            rew_list.append(r)
            obs_list.append(s)
            act_list.append(a)
            logp_list.append(
                log_prob if torch.is_tensor(log_prob) else torch.tensor(log_prob)
            )
        return obs_list, act_list, rew_list, logp_list

    def _gae(self, obs_traj, rew_traj, last_val=0.0):
        """
        Compute GAE-λ advantages and rewards-to-go for a single trajectory.
        Values V(s) are estimated by the current critic (no gradient).
        """
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_traj, dtype=torch.float32)
            vals  = self.pol.value(obs_t).squeeze(-1).numpy()

        rews = rew_traj + [last_val]
        vals = list(vals) + [last_val]

        advantages, returns = [], []
        gae = 0.0
        for t in reversed(range(len(rew_traj))):
            delta = rews[t] + self.gamma * vals[t + 1] - vals[t]
            gae   = delta + self.gamma * self.lam * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + vals[t])   # advantage + baseline = return

        return advantages, returns

    def _evaluate(self, obs_t, act_t):
        """
        Recompute log π(a|s), V(s), entropy under the *current* policy.
        Called inside the update loop — gradients are ON.
        """
        out = self.pol(obs_t)

        if self.discrete:
            dist     = torch.distributions.Categorical(out)
            log_prob = dist.log_prob(act_t)
        else:
            mu, std  = out[0], out[1]
            dist     = torch.distributions.Normal(mu, std)
            log_prob = dist.log_prob(act_t).sum(axis=-1)

        entropy = dist.entropy().mean()
        v       = self.pol.value(obs_t).squeeze(-1)

        return log_prob, v, entropy

    def _as_action_tensor(self, act_list):
        """Convert a list of numpy arrays / tensors into a single tensor."""
        return torch.stack([torch.as_tensor(a) for a in act_list])
