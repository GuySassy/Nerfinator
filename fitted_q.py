import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer

def Fitted_Q_iteration(encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma, learning_rate):
        next_q_values = np.empty([len(rewards), 3])
        for i in range(len(rewards)):
            action_vector = np.array(range(3))
            next_state_vector = np.tile(encoded_next_states[i, :], (3,1))
            next_q = linear_policy.get_q_values(next_state_vector, action_vector)
            if done_flags[i]:
                 next_q = np.ones((3, 1)) * rewards[i]
            next_q_values[i, :] = next_q.flatten()


        encoded_states_tiled = linear_policy.get_q_features(encoded_states, actions)
        targets = rewards + gamma * np.max(next_q_values, axis=1)
        targets = np.expand_dims(targets, axis=1)
        q = linear_policy.get_q_values(encoded_states,actions)
        is_equal = np.all(encoded_states == encoded_next_states, axis=1)
        # targets[is_equal] = 1
        # q[is_equal] = 1
        targets[done_flags] = 1
        q[done_flags] = 1
        bellman_err = targets - q
        MSE = (1/len(rewards)) * bellman_err.T @ bellman_err
        grad = (2/len(rewards)) * encoded_states_tiled.T @ bellman_err
        w = linear_policy.w
        new_w = w + learning_rate * grad
        return new_w, MSE


if __name__ == '__main__':
    samples_to_collect = 100000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.999
    w_updates = 100
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000

    np.random.seed(123)
    # np.random.seed(234)

    env = MountainCarWithResetEnv()
    # collect data
    states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    for i in range(len(done_flags)):
        if states[i][0] >= env.goal_position:
            next_states[i] = states[i]
            rewards[i] = 1
        # if states[i][0] < env.goal_position and done_flags[i]:
        #     rewards[i] = 0
    # get data success rate
    data_success_rate = np.sum(rewards) / len(rewards)
    print(f'success rate {data_success_rate}')
    # # standardize data
    data_transformer = DataTransformer()
    data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    states = data_transformer.transform_states(states)
    next_states = data_transformer.transform_states(next_states)
    # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # # encode all states:
    encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)
    encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, True)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    # q_features = linear_policy.get_q_features(encoded_states, actions)
    # start an object that evaluates the success rate over time
    evaluator = GamePlayer(env, data_transformer, feature_extractor, linear_policy)
    error = []
    for fitted_q_iteration in range(w_updates):
        print(f'starting lspi iteration {fitted_q_iteration}')
        new_w, err = Fitted_Q_iteration(
            encoded_states, encoded_next_states, actions, rewards, done_flags, evaluator.policy, gamma, learning_rate=0.1
        )
        norm_diff = evaluator.policy.set_w(new_w)
        if norm_diff < 0.00001:
            break
        error.append(np.squeeze(err))
    print('done lspi')
    evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)
    # plot the error
    plt.plot(error)

    # Label the axes
    plt.xlabel("Index")
    plt.ylabel("Error")

    # Set the title of the plot
    plt.title("Error Plot")

    # Show the plot
    plt.show()




