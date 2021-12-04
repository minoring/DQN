import numpy as np
import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:
    def __init__(self, replay_memory_size, agent_history_length):
        self.memory = deque([], maxlen=replay_memory_size)
        self.frame_queue = FrameQueue(agent_history_length)
        self.heldout_states = None  # Should be sample from replay memory.

    def push(self, frame, action, next_frame, reward):
        """Save transition"""
        if self.frame_queue.filled():
            state = self.frame_queue.stack()
            if next_frame is None:
                next_state = None
            else:
                self.frame_queue.push(next_frame)
                next_state = self.frame_queue.stack()
            self.memory.append(Transition(state, action, next_state, reward))
        else:
            self.frame_queue.push(frame)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_heldout_states(self, heldout_states_size):
        if self.heldout_states is None:
            # If it is None, heldout states are never defined, we sample from our memory.
            sample = self.sample(heldout_states_size)
            batch = Transition(*zip(*sample))
            self.heldout_states = batch.state
        assert len(self.heldout_states) == heldout_states_size
        return self.heldout_states

    def __len__(self):
        return len(self.memory)


class FrameQueue:
    """Queue to store m most recent frames.
    This agent history length in DQN paper.
    """
    def __init__(self, his_len):
        self.his_len = his_len
        self.queue = deque([], maxlen=self.his_len)

    def push(self, frame):
        self.queue.append(frame)

    def filled(self):
        return len(self.queue) == self.his_len

    def clear(self):
        """Frame should be cleared after the end of an episode."""
        self.queue.clear()

    def stack(self):
        """Stack m most recent frames."""
        assert self.filled()
        return np.stack(self.queue)
