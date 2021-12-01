import gym

from testrun_wrapper import TestRunWrapper


def get_default_env_by_name(env_name, render_mode=None):
    # Use CartPole-v0 for the test run
    if env_name == 'CartPole-v0':
        return TestRunWrapper(gym.make(env_name))

    env = gym.make(
        env_name,
        obs_type='rgb',  # ram | rgb | grayscale
        frameskip=1,  # frame skip
        mode=0,  # game mode, see Machado et al. 2018
        difficulty=0,  # game difficulty, see Machado et al. 2018
        repeat_action_probability=0.,  # Sticky action probability
        full_action_space=True,  # Use all action
        render_mode=render_mode  # None | 'human' | 'rgb_array'
    )

    return env
