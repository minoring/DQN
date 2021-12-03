import cv2
import gym
import numpy as np
from gym.spaces import Box


class AtariPreprocessing(gym.Wrapper):
    """Preprocessed Atari 2601 environment.

    Reference:
        - https://github.com/openai/gym/blob/master/gym/wrappers/atari_preprocessing.py
        - https://github.com/google/dopamine/blob/df97ba1b0d4edf90824534efcdda20d6549c37a9/dopamine/discrete_domains/atari_lib.py#L329-L515
    """
    def __init__(self, env, config, is_training):
        super().__init__(env)
        self.env = env
        self.config = config
        self.frame_skip = self.config['env']['frame_skip']

        self.obs_buffer = [
            np.empty(env.observation_space.shape[:2], dtype=np.uint8),  # (210, 160)
            np.empty(env.observation_space.shape[:2], dtype=np.uint8),
        ]

        # Test run does not have ale.
        self.ale = env.unwrapped.ale if hasattr(env.unwrapped, 'ale') else None

        self.lives = 0  # Will need to be set by reset().
        self.is_training = is_training

        self.screen_size = self.config['env']['screen_size']
        self.obs_shape = (self.screen_size, self.screen_size)
        self.observation_space = Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.ep_reward = 0  # Cummulative reward in this eposide.

    def step(self, action):
        R = 0.
        for t in range(1, self.frame_skip + 1):
            _, reward, done, info = self.env.step(action)
            R += reward
            if done:
                break
            if t == self.frame_skip - 1:
                self.obs_buffer[1] = self._get_screen_grayscale()
            elif t == self.frame_skip:
                self.obs_buffer[0] = self._get_screen_grayscale()
        if self.is_training:
            R = AtariPreprocessing._clip_reward(R)

        self.ep_reward += R
        return self._pool_and_resize(), R, done, info

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        self.ep_reward = 0
        # TODO(minho): Reset on live loss?
        # self.lives = self.ale.lives()
        # TODO(minho): Implement NOOP?

        self.obs_buffer[0] = self._get_screen_grayscale()
        self.obs_buffer[1].fill(0)
        return self._pool_and_resize()

    def _pool_and_resize(self):
        if self.frame_skip > 1:
            np.maximum(self.obs_buffer[0], self.obs_buffer[1], out=self.obs_buffer[0])

        resized_img = cv2.resize(self.obs_buffer[0], (self.screen_size, self.screen_size),
                                 interpolation=cv2.INTER_AREA)
        return resized_img

    def _clip_reward(reward):
        if reward > 0:
            return 1
        elif reward < 0:
            return -1
        return 0

    def _get_screen_grayscale(self):
        if self.ale is not None:
            screen = np.empty(self.env.observation_space.shape[:2], dtype=np.uint8)
            self.ale.getScreenGrayscale(screen)
        else:
            screen = self.env.render(mode='rgb_array')
            screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

        assert screen is not None
        return screen
