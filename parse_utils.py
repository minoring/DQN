import argparse


def get_args():
    parser = argparse.ArgumentParser("Train DQN playing Atari")
    parser.add_argument('--config', '-c', help='Path to config file', default='config.yaml')
    parser.add_argument('--env', '-e', help='Environment of RL algorithm', required=True)
    parser.add_argument('--log-interval',
                        help='How many steps to wait to log',
                        type=int,
                        default=100)
    parser.add_argument('--log-csv-path', help='Path to csv file to save log', default='log.csv')
    parser.add_argument('--log-dir', help='Directory to save log summary', default='log')
    parser.add_argument('--model-save-path', help='Path to save trained model', default='model.pt')

    args = parser.parse_args()

    return args
