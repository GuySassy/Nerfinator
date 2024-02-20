import argparse
import torch
from agent import DataCollector, DataTransformer, Policy, AimBotWithResetEnv, RadialBasisFunctionExtractor, GamePlayer
import numpy as np
import math
from functools import partial
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_arguments():
    parser = argparse.ArgumentParser(description='Script Description')
    parser.add_argument('--perc_train', type=float, default=1, help='percentage of train data to use for the entire experiment (can be used to run experiments with reduced datasets to test small data scenarios)')
    parser.add_argument('--samples', type=int, default=300, help='Number of samples to collect from env')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--train_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--train', type=str2bool, nargs='?', const=True, default=True, help='To train a model ?')
    parser.add_argument('--lr', type=float, default=1e-3, help='Maximum learning rate')
    # Add more arguments here as needed

    return parser.parse_args()


if __name__ == '__main__':
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {DEVICE}")
    args = parse_arguments()
    BATCH_SIZE = args.batch_size
    TRAIN_EPOCHS = args.train_epochs
    SAMPLES_NUM = args.samples
    DEBUG = 1
    if DEBUG:
        samples_to_collect = SAMPLES_NUM
        number_of_kernels_per_dim = [12, 10, 12, 10]
        gamma = 0.99
        evaluation_number_of_games = 10
        evaluation_max_steps_per_game = 1000

        np.random.seed(123)
        env = AimBotWithResetEnv(resolution=(480, 640), accuracy=4)
        # # collect data
        states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
        # # get data success rate
        data_success_rate = np.sum(rewards) / len(rewards)
        print(f'success rate {data_success_rate}')
        # # standardize data
        data_transformer = DataTransformer()
        # data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
        # states = data_transformer.transform_states(states)
        # next_states = data_transformer.transform_states(next_states)
        # # process with radial basis functions
        feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        W = np.random.uniform(size=4)
        print(W)
        # policy , action = softmax_policy(linear_policy.w, feature_extractor, encoded_states[1, :], alpha=1)
        # print(action)
        # grad = log_softmax_gradient(linear_policy.w, feature_extractor, alpha=1, encoded_states=encoded_states[1,:], action=0)
        # print(grad)
        # # start an object that evaluates the success rate over time
        policy = Policy(env, data_transformer, feature_extractor, W, epsilon=0.3)
        # state = env.reset()
        # evaluator.play_game(evaluation_max_steps_per_game, start_state=state)
        success_rate = []
        for pol_grad_iteration in range(TRAIN_EPOCHS):
            print(f'starting policy gradient iteration {pol_grad_iteration}')

            new_W, states = policy.policy_gradient_iteration(feature_extractor, alpha=1,
                                                             tau=evaluation_max_steps_per_game, epsilon=0.001)
            # linear_policy.set_w(new_W)
            # new_w = policy_gradient_iteration(
            #     encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
            # )
            norm_diff = policy.set_w(new_W)
            if norm_diff < 0.00001:
                break
            data_transformer.set_using_states(states)
            states = data_transformer.transform_states(states)
            evaluator = GamePlayer(env, data_transformer, feature_extractor, policy)
            success_rate.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
        print('done policy gradient')
        evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
        evaluator.play_game(evaluation_max_steps_per_game, render=True)

# if __name__ == '__main__':
#     samples_to_collect = 100000
#     # samples_to_collect = 150000
#     # samples_to_collect = 10000
#     number_of_kernels_per_dim = [12, 10, 12, 10]
#     gamma = 0.99
#     w_updates = 100
#     evaluation_number_of_games = 10
#     evaluation_max_steps_per_game = 1000
#
#     np.random.seed(123)
#     # np.random.seed(234)
#
#     env = AimBotWithResetEnv((480, 640), 4)
#     # # collect data
#     # states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
#     # # get data success rate
#     # data_success_rate = np.sum(rewards) / len(rewards)
#     # print(f'success rate {data_success_rate}')
#     # # standardize data
#     data_transformer = DataTransformer()
#     # data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
#     # states = data_transformer.transform_states(states)
#     # next_states = data_transformer.transform_states(next_states)
#     # # process with radial basis functions
#     feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
#     W = np.random.uniform(size=4)
#     print(W)
#     # policy , action = softmax_policy(linear_policy.w, feature_extractor, encoded_states[1, :], alpha=1)
#     # print(action)
#     # grad = log_softmax_gradient(linear_policy.w, feature_extractor, alpha=1, encoded_states=encoded_states[1,:], action=0)
#     # print(grad)
#     # # start an object that evaluates the success rate over time
#     policy = Policy(env, data_transformer, feature_extractor,  W, epsilon=0.3)
#     # state = env.reset()
#     # evaluator.play_game(evaluation_max_steps_per_game, start_state=state)
#     success_rate = []
#     for pol_grad_iteration in range(w_updates):
#         print(f'starting policy gradient iteration {pol_grad_iteration}')
#
#         new_W, states = policy.policy_gradient_iteration(feature_extractor, alpha=1, tau=evaluation_max_steps_per_game, epsilon=0.001)
#         # linear_policy.set_w(new_W)
#         # new_w = policy_gradient_iteration(
#         #     encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
#         # )
#         norm_diff = policy.set_w(new_W)
#         if norm_diff < 0.00001:
#             break
#         data_transformer.set_using_states(states)
#         states = data_transformer.transform_states(states)
#         evaluator = GamePlayer(env, data_transformer, feature_extractor, policy)
#         success_rate.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
#     print('done policy gradient')
#     evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
#     evaluator.play_game(evaluation_max_steps_per_game, render=True)

