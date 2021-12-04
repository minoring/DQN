import random

import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_utils as ptu


class DQN(nn.Module):
    def __init__(self, config, action_space):
        super(DQN, self).__init__()

        self.config = config
        self.action_space = action_space

        self.conv1 = nn.Conv2d(config['agent_history_length'], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        width = self._compute_conv2d_size_out(self.config['env']['screen_size'], 8, 4)
        width = self._compute_conv2d_size_out(width, 4, 2)
        width = self._compute_conv2d_size_out(width, 3, 1)
        height = width  # We use square kernel and square image.
        linear_input_size = height * width * 64

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(linear_input_size, 512)
        self.output = nn.Linear(512, self.action_space)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.output(x)

    def eps_action_selection(self, frame_idx, frame_queue):
        eps = self.compute_epsilon(frame_idx)
        if random.random() <= eps or not frame_queue.filled():
            # We take exploratory action with epsilon probability or
            # we can not create state since frame queue is not filled.
            return random.randrange(self.action_space)
        # Greedy action selection
        with torch.no_grad():
            state = ptu.from_img(frame_queue.stack()).unsqueeze(0)
            return torch.argmax(self(state)).item()

    def compute_epsilon(self, frame_idx):
        if frame_idx > self.config['exploration']['final_exploration_frame']:
            eps = self.config['exploration']['final_exploration']
        else:
            init_eps = self.config['exploration']['initial_exploration']
            fin_eps = self.config['exploration']['final_exploration']
            fin_expr_frame = self.config['exploration']['final_exploration_frame']

            eps = init_eps - (init_eps - fin_eps) / fin_expr_frame * frame_idx
        return eps

    def _compute_conv2d_size_out(self, input_size, kernel_size, stride):
        return (input_size - (kernel_size - 1) - 1) // stride + 1
