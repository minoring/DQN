import random

import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

import pytorch_utils as ptu
from pytorch_utils import device
from atari_preprocessing import AtariPreprocessing
from parse_utils import get_args
from env_utils import get_default_env_by_name
from dqn import DQN
from replay_memory import ReplayMemory, Transition
from logger import Logger


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger = Logger(args.env, args.log_csv_path, args.log_dir)

    env = get_default_env_by_name(args.env)
    env = AtariPreprocessing(env, config, is_training=True)

    # Set random seeds.
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    agent = DQN(config, env.action_space.n)
    target_network = DQN(config, env.action_space.n)
    target_network.load_state_dict(agent.state_dict())

    agent.to(device)
    target_network.to(device)

    memory = ReplayMemory(config['replay_memory_size'], config['agent_history_length'])
    criterion = nn.HuberLoss(reduction='sum')
    # optimizer = optim.RMSprop(agent.parameters(),
    #                           lr=config['train']['learning_rate'],
    #                           momentum=config['train']['gradient_momentum'])
    optimizer = optim.Adam(agent.parameters(), lr=config['train']['learning_rate'])
    frame = env.reset()
    done = False
    num_updates = 0
    for frame_idx in range(1, config['train']['total_frame'] + 1):
        # A uniform random policy is run for this number of frames before learning starts
        if frame_idx < config['train']['replay_start_size']:
            action = env.action_space.sample()
        else:
            action = agent.eps_action_selection(frame_idx, memory.frame_queue)

        next_frame, reward, done, _ = env.step(action)
        if done:
            next_frame = None
        memory.push(frame, action, next_frame, reward)
        frame = next_frame
        if done:
            logger.log_ep_reward(env.ep_reward)
            frame = env.reset()
            memory.frame_queue.clear()
        if frame_idx < config['train']['replay_start_size']:
            # No training in this case
            continue

        sample = memory.sample(config['train']['minibatch_size'])
        # Batch of transition to Transition of batch.
        # (B, Transition) -> Transition(B state, B action, B next_state, B reward)
        batch = Transition(*zip(*sample))

        non_terminal_next_state_mask = torch.tensor([s is not None for s in batch.next_state],
                                                    device=device,
                                                    dtype=torch.bool)
        non_terminal_next_state = ptu.from_img(
            np.stack([s for s in batch.next_state if s is not None]))

        state = ptu.from_img(np.array(batch.state))
        action = torch.tensor(batch.action).to(device).unsqueeze(-1)
        action_value = agent(state).gather(1, action)

        target = ptu.from_tuple(batch.reward)
        target_q = target_network(non_terminal_next_state).max(dim=1).values.detach()
        target[non_terminal_next_state_mask] += config['train']['discount_factor'] * target_q

        loss = criterion(action_value, target.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        num_updates += 1

        if num_updates % config['train']['target_network_update_frequency'] == 0:
            target_network.load_state_dict(agent.state_dict())

        if num_updates % args.log_interval == 0:
            total_frame = config['train']['total_frame'] + 1
            print("frame {}/{} ({:.2f}%), loss: {:.6f}".format(frame_idx, total_frame,
                                                               frame_idx / total_frame * 100.,
                                                               loss.item()))

    logger.save()
    torch.save(agent.state_dict(), args.model_save_path)
    env.close()


if __name__ == '__main__':
    main()
