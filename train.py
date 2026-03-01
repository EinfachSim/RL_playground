import copy
import wandb
from utils import load_session
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

    wandb.init(project="RL_playground", config=config)

    train_episode = 1
    try:
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

            rewards.append(total_reward)
            batches.append(trajectory)
            if len(batches) == batch_size:
                avg_reward = sum(rewards[-batch_size:])/batch_size
                print(f"EPISODE: {train_episode}, REWARD: {avg_reward}")
                ### LOGGING
                wandb.log({"reward": avg_reward, "best_reward": best}, step=train_episode)
                ####
                agent.update(batches)
                batches = []
                train_episode += 1
    except KeyboardInterrupt:
        print("\nTraining interrupted, saving best model to disk...")
    
    finally:
        env.close()
        wandb.finish()
        agent.pol = best_policy
        with open("best_agent.pkl", "wb") as f:
            pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)


    env.close()
    wandb.finish()

    #save best agent

    agent.pol = best_policy
    with open("best_agent.pkl", "wb") as f:
        pickle.dump(agent, f, pickle.HIGHEST_PROTOCOL)
    
    return rewards