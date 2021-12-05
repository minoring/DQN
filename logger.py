import os
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

import pytorch_utils as ptu


class Logger:
    def __init__(self, env_name, log_csv_path, log_dir, split='train'):
        self.env_name = env_name
        self.log_csv_path = log_csv_path
        self.split = split
        self.ep = 0
        self.ep_rewards = []
        self.losses = []

        self.log_dir = os.path.join(log_dir, env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.writer = SummaryWriter(self.log_dir)

    def log_ep_reward(self, reward):
        self.ep_rewards.append(reward)
        self.ep += 1
        self.writer.add_scalar(f'EposideReward/{self.split}', reward, self.ep)

    def log_scalar(self, name, scalar, step):
        self.writer.add_scalar(f'{name}/{self.split}', scalar, step)

    def log_q(self, agent, heldout_states, minibatch_size, step):
        with torch.no_grad():
            state_shape = heldout_states[0].shape
            minibatch_heldout_states = np.array(heldout_states).reshape(-1, minibatch_size,
                                                                        *state_shape)
            accum_max_q = 0
            accum_min_q = 0
            for heldout_state in minibatch_heldout_states:
                Q = agent(ptu.from_img(heldout_state))
                accum_max_q += Q.max(dim=1).values.sum().item()
                accum_min_q += Q.min(dim=1).values.sum().item()

            average_max_q = accum_max_q / len(heldout_states)
            average_min_q = accum_min_q / len(heldout_states)
            self.log_scalar('AverageMaxQ', average_max_q, step)
            self.log_scalar('AverageMinQ', average_min_q, step)

    def save(self):
        split = [self.split] * self.ep
        env_names = [self.env_name] * self.ep
        ep_idxs = list(range(1, self.ep + 1))
        df = pd.DataFrame(list(zip(split, env_names, ep_idxs, self.ep_rewards)))

        df.to_csv(self.log_csv_path, mode='a')
        print(f'Log saved at {self.log_csv_path}')
