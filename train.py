import copy
from utils import load_session
from logger import HTTPLogger
import pickle


def train_from_config(config_file):
    env, agent, config = load_session(config_file)

    obs, _ = env.reset()

    num_episodes = config["num_episodes"]
    batch_size = config["batch_size"]

    rewards = []
    batches = []

    best = -float("Inf")
    best_policy = None

    logger = HTTPLogger()

    train_episode = 1

    for e in range(num_episodes*batch_size):

        obs, _ = env.reset()

        total_reward = 0
        done = False
        trajectory = []

        while not done:

            action, log_prob = agent.sample_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            trajectory.append((reward, obs, action, log_prob))

            obs = next_obs

            total_reward += reward

            done = terminated or truncated

        if total_reward > best:
            best_policy = copy.deepcopy(agent.pol)
            best = total_reward

        batches.append(trajectory)
        batches = batches[-batch_size:]
        if e % batch_size == 0:
            print(f"EPISODE: {train_episode}, REWARD: {total_reward}")
            ### LOGGING
            log_data = {
                "episode": train_episode,
                "reward": total_reward
            }
            logger.log_episode_metrics(log_data)
            ####
            agent.update(batches)
            train_episode += 1

            rewards.append(total_reward)

    env.close()

    #save best agent

    agent.pol = best_policy
    with open("best_agent.pkl", "wb") as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
    
    return rewards