import yaml

def load_session(filename, render_mode=None):
    with open(filename, "r") as f:
        config = yaml.safe_load(f)

    env_config = config["env"]
    module = __import__("envs")
    class_ = getattr(module, env_config["name"])
    env = class_(render=render_mode)

    module = __import__("algos")
    class_ = getattr(module, config["algo"])
    agent = class_(config["model_params"])
    
    return env, agent, config
