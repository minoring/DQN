import yaml
import torch
import pandas as pd
from gym.wrappers import RecordVideo

from atari_preprocessing import AtariPreprocessing
from dqn import DQN
from env_utils import get_default_env_by_name
from parse_utils import get_test_args
from pytorch_utils import device
import pytorch_utils as ptu
from replay_memory import FrameQueue


def save_csv(episode_rewards, env_name, log_csv_path):
    num_ep = len(episode_rewards)
    split = ['test'] * num_ep
    env_names = [env_name] * num_ep
    ep_idxs = list(range(1, num_ep + 1))
    df = pd.DataFrame(list(zip(split, env_names, ep_idxs, episode_rewards)))

    df.to_csv(log_csv_path, mode='a')
    print(f'Log saved at {log_csv_path}')


def main():
    args = get_test_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    env = get_default_env_by_name(args.env)
    env = AtariPreprocessing(env, config, is_training=False)
    agent = DQN(config, env.action_space.n)
    agent.load_state_dict(torch.load(args.trained_model_path))
    agent.to(device)

    frame_queue = FrameQueue(config['agent_history_length'])
    env.reset()

    epsiode_rewards = []
    for test_run in range(args.num_test_run):
        # Save video if it is last test run.
        if args.record_video and test_run == args.num_test_run - 1:
            video_save_folder = f'{args.env}_video'
            env = RecordVideo(env, video_save_folder)
            print(f'video saved at: {video_save_folder}')

        # Fill out frame queue before run.
        for _ in range(config['agent_history_length']):
            action = env.action_space.sample()
            frame, _, _, _ = env.step(action)
            frame_queue.push(frame)

        done = False
        episode_reward = 0
        while not done:
            state = ptu.from_img(frame_queue.stack()).unsqueeze(0)
            action = agent(state).argmax().item()
            frame, reward, done, _ = env.step(action)

            episode_reward += reward
            frame_queue.push(frame)

        print(f"Episode reward test run {test_run}: {episode_reward}")
        epsiode_rewards.append(episode_reward)
        env.reset()
        frame_queue.clear()
    save_csv(epsiode_rewards, args.env, args.log_csv_path)


if __name__ == '__main__':
    main()
