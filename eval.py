import pickle
from utils import load_session

#Evaluation
with open("best_agent.pkl", "rb") as f:
    best_agent = pickle.load(f)
    best_agent.eval = True

env, _, config = load_session("config.yaml", render_mode="human")
obs, _ = env.reset()

done = False
total_reward = 0

while not done:

    action = best_agent.sample_action(obs)
    next_obs, reward, terminated, truncated, info = env.step(action)

    obs = next_obs

    total_reward += reward

    done = terminated or truncated

env.close()