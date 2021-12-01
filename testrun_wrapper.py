import cv2
import gym
import numpy as np
from gym.spaces import Box


class TestRunWrapper(gym.Wrapper):
    """Wrap test environment to return rgb observation"""
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        screen_shape = (400, 600)  # TODO(minho): Hard coding for now.
        self.observation_space = Box(low=0, high=255, shape=screen_shape, dtype=np.uint8)

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        observation = self._get_screen()
        return observation, reward, done, info

    def _get_screen(self):
        screen = self.env.render('rgb_array')
        screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)
        return screen
