# RL Playground

A clean, modular implementation of Deep Reinforcement Learning algorithms built from scratch in PyTorch, loosely following [OpenAI's Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/). Designed as a personal playground for experimenting with RL algorithms and meta-learning approaches.

---

## Algorithms

| Algorithm | Type | Status |
|-----------|------|--------|
| REINFORCE | Policy Gradient | ✅ |

More algorithms coming soon.

---

## Project Structure

```
RL_playground/
├── main.py          # Entry point — launches training
├── train.py         # Training loop
├── eval.py          # Evaluation — renders the best saved policy
├── algos.py         # RL algorithm implementations (REINFORCE, ...)
├── agents.py        # Neural network policy (MLPAgent)
├── envs.py          # Modular environment wrappers (Gymnasium-compatible)
├── utils.py         # Session loading helpers
├── config.yaml      # Experiment configuration
└── requirements.txt
```

---

## Setup

Tested on Python 3.10.15 (M3 MacBook Air).

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Training

Configure your experiment in `config.yaml`, then run:

```bash
python main.py
```

Training saves the best-performing policy to `best_agent.pkl`.

### Evaluation

Renders the saved policy in the environment:

```bash
python eval.py
```

---

## Configuration

All experiment parameters are set in `config.yaml`:

```yaml
env:
  name: ParamCartPole       # Environment to use

num_episodes: 1000          # Number of training episodes
batch_size: 10              # Episodes per policy update

algo: REINFORCE             # Algorithm to use

model_params:
  hidden_sizes: [128, 32]   # MLP hidden layer sizes
  init_log_std: 0.7         # Initial log std (continuous actions)
  learning_rate: 3e-4       # Adam learning rate
  discr: True               # True for discrete, False for continuous action spaces
  in_dim: 4                 # Observation space dimension
  out_dim: 2                # Action space dimension
```

---

## Environments

Environments live in `envs.py` and are built to be modular and plug-and-play. Any environment implementing the [Gymnasium](https://gymnasium.farama.org/) interface can be dropped in. The parameterized structure also lays the groundwork for meta-learning experiments where environment parameters are randomized across episodes.

---

## Monitoring

Training is monitored in real time via [Weights & Biases](https://wandb.ai). After setup, a live dashboard link is printed to the terminal at the start of each run.

```bash
pip install wandb
wandb login    # one-time setup
```

Metrics logged per update: `reward`, `best_reward`.

---

## Agent Architecture

`MLPAgent` is a configurable MLP policy supporting both discrete and continuous action spaces:

- **Discrete:** outputs a softmax over actions (Categorical policy)
- **Continuous:** outputs `(mu, std)` for a Normal distribution; `log_std` is a learnable parameter

---

## Roadmap

- [ ] PPO
- [ ] SAC
- [ ] Meta-learning (MAML / RL²)
- [ ] Multi-environment benchmarking

---

## References

- [Spinning Up in Deep RL — OpenAI](https://spinningup.openai.com/en/latest/)
- [Gymnasium](https://gymnasium.farama.org/)