import argparse


def get_train_args():
    parser = argparse.ArgumentParser("Train DQN playing Atari")
    parser.add_argument('--seed', help='random seed (default: 1)', default=1)
    parser.add_argument('--config', '-c', help='Path to config file', default='config.yaml')
    parser.add_argument('--env', '-e', help='Environment of RL algorithm', required=True)
    parser.add_argument('--log-interval',
                        help='How many steps to wait to log',
                        type=int,
                        default=1000)
    parser.add_argument('--log-csv-path', help='Path to csv file to save log', default='log.csv')
    parser.add_argument('--log-dir', help='Directory to save log summary', default='log')
    parser.add_argument('--model-save-path', help='Path to save trained model', default='model.pt')

    args = parser.parse_args()

    return args


def get_test_args():
    parser = argparse.ArgumentParser("Test DQN playing Atari")
    parser.add_argument('--trained-model-path',
                        help='Path to trained model. Use this model for testing',
                        required=True)
    parser.add_argument('--env', '-e', help='Environment of RL algorithm', required=True)
    parser.add_argument('--num-test-run', help='Number of test run', type=int, default=10)
    parser.add_argument('--config', '-c', help='Path to config file', default='config.yaml')
    parser.add_argument('--record-video',
                        help='Wheter to save video files',
                        action='store_true')
    parser.add_argument('--log-csv-path', help='Path to csv file to save log', default='log.csv')

    args = parser.parse_args()

    return args
