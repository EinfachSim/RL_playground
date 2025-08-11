import gymnasium as gym
import numpy as np

class ParamAcrobot(gym.Wrapper):
    def __init__(self, render=None):
        env = gym.make("Acrobot-v1", render_mode=render)
        super().__init__(env)

class ParamMountainCarCont(gym.Wrapper):
    def __init__(self, render=None):
        env = gym.make("MountainCarContinuous-v0", render_mode=render)
        super().__init__(env)

class ParamBipedalWalker(gym.Wrapper):
    def __init__(self, render=None):
        env = gym.make("BipedalWalker-v3", hardcore=False, render_mode=render)
        super().__init__(env)

class ParamLunarLander(gym.Wrapper):
    def __init__(self, render=None):
        env = gym.make("LunarLander-v3", render_mode=render)
        super().__init__(env)




class ParamCartPole(gym.Wrapper):
    def __init__(self, render=None, gravity=9.8, pole_length=0.5, cart_mass=1.0, force_mag=10.0):
        env = gym.make("CartPole-v1", render_mode=render)
        super().__init__(env)
        
        # Store parameters
        self.gravity = gravity
        self.pole_length = pole_length
        self.cart_mass = cart_mass
        self.force_mag = force_mag
        
        # Modify environment physics
        self.env.unwrapped.gravity = self.gravity
        self.env.unwrapped.length = self.pole_length
        self.env.unwrapped.masscart = self.cart_mass
        self.env.unwrapped.force_mag = self.force_mag
        
        # Recompute dependent values
        self.env.unwrapped.total_mass = self.cart_mass + self.env.unwrapped.masspole
        self.env.unwrapped.polemass_length = self.env.unwrapped.masspole * self.pole_length

    @staticmethod
    def sample_task():
        """Randomly sample environment parameters for a new task."""
        return {
            "gravity": np.random.uniform(8.0, 12.0),
            "pole_length": np.random.uniform(0.4, 0.8),
            "cart_mass": np.random.uniform(0.5, 1.5),
            "force_mag": np.random.uniform(8.0, 12.0)
        }

    @classmethod
    def make_with_random_params(cls, render=None):
        """Factory method to create a ParamCartPole with random parameters."""
        params = cls.sample_task()
        instance = cls(render, **params)
        return instance, instance.params
