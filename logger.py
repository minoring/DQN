import os
import time

import pandas as pd
from torch.utils.tensorboard import SummaryWriter


class Logger:

    def __init__(self, env_name, log_csv_path, log_dir):
        self.env_name = env_name
        self.log_csv_path = log_csv_path
        self.ep = 0
        self.ep_rewards = []

        self.log_dir = os.path.join(log_dir, env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S"))
        self.writer = SummaryWriter(self.log_dir)

    def log_ep_reward(self, reward):
        self.ep_rewards.append(reward)
        self.ep += 1
        self.writer.add_scalar('EposideReward/train', reward, self.ep)

    def save(self):
        env_names = [self.env_name] * self.ep
        ep_idxs = list(range(1, self.ep))
        df = pd.DataFrame(list(
            zip(env_names, ep_idxs, self.ep_rewards)
        ))

        df.to_csv(self.log_csv_path, mode='w')
        print(f'Log saved at {self.log_csv_path}')
