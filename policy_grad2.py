import numpy as np
import time
import matplotlib.pyplot as plt 

from data_transformer import DataTransformer
from mountain_car_with_data_collection import MountainCarWithResetEnv
from radial_basis_function_extractor import RadialBasisFunctionExtractor


class Solver:
    def __init__(self, number_of_kernels_per_dim, number_of_actions, learning_rate):
        # Set max value for normalization of inputs
        self._max_normal = 1
        # get state \action information
        self.data_transformer = DataTransformer()
        state_mean = [-3.00283763e-01,  5.61618575e-05]
        state_std = [0.51981243, 0.04024895]
        self.data_transformer.set(state_mean, state_std)
        self._actions = number_of_actions
        # create RBF features:
        self.feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
        self.number_of_features = self.feature_extractor.get_number_of_features()
        # the weights of the q learner
        self.theta = np.random.uniform(-0.001, 0, size=number_of_actions * self.number_of_features)
        self.learning_rate = learning_rate

    def _normalize_state(self, s):
        return self.data_transformer.transform_states(np.array([s]))[0]

    def get_features(self, state):
        normalized_state = self._normalize_state(state)
        features = self.feature_extractor.encode_states_with_radial_basis_functions([normalized_state])[0]
        return features

    def get_state_action_features(self, state, action):
        state_features = self.get_features(state)
        all_features = np.zeros(len(state_features) * self._actions)
        all_features[action * len(state_features): (1 + action) * len(state_features)] = state_features
        return all_features
    
    def softmax_policy(self, state, alpha):
        array = []
        sum = 0 
        for i in range(3):
            encoded_states_tiled = self.get_state_action_features(state, i)
            up = np.exp(alpha*encoded_states_tiled@self.theta)
            array.append(up)
            sum += up

        policy = array / sum
        action = np.argmax(policy)
        return policy, action
    
    def log_softmax_gradient(self, state, action, alpha):
        encoded_states = self.get_features(state)
        up_sum = np.zeros(encoded_states.shape[0] * 3) 
        down_sum = 0 # dim is 1 (exp of inner prod) 
        for i in range(3):
            encoded_states_tiled = self.get_state_action_features(state, i)
            A = alpha*encoded_states_tiled.T * np.exp(alpha*encoded_states_tiled@self.theta)
            A=A.reshape(-1)
            up_sum += A
            down_sum += np.exp(alpha*encoded_states_tiled@self.theta)
        b = up_sum / down_sum
        a = alpha * self.get_state_action_features(state, action)
        return a - b


    def update_theta(self, grad_sum, rewards_sum):
        grad = rewards_sum * grad_sum
        #updating the weights
        self.theta = self.theta + self.learning_rate * grad.T


def modify_reward(reward):
    reward -= 1
    if reward == 0.:
        reward = 100.
    return reward


def run_episode(env, solver, alpha, is_train=True, epsilon=None, max_steps=200, render=False):
    episode_gain = 0
    grad_gain = 0
    if is_train:
        start_position = np.random.uniform(env.min_position, env.goal_position - 0.01)
        start_velocity = np.random.uniform(-env.max_speed, env.max_speed)
    else:
        start_position = -0.5
        start_velocity = np.random.uniform(-env.max_speed / 100., env.max_speed / 100.)
    state = env.reset_specific(start_position, start_velocity)
    step = 0
    if render:
        env.render()
        time.sleep(0.1)
    while True:
        if epsilon is not None and np.random.uniform() < epsilon:
            action = np.random.choice(env.action_space.n)
        else:
            _, action = solver.softmax_policy(state, alpha)
        if render:
            env.render()
            time.sleep(0.1)
        next_state, reward, done, _ = env.step(action)
        reward = modify_reward(reward)
        step += 1
        episode_gain += reward
        grad_gain += solver.log_softmax_gradient(state, action, alpha=1)
        if done or step == max_steps:
            break
        state = next_state
    if is_train:
        solver.update_theta(grad_gain, episode_gain)
    return episode_gain


if __name__ == "__main__":
    env = MountainCarWithResetEnv()
    seed = 123
    # seed = 234
    # seed = 345
    np.random.seed(seed)
    env.seed(seed)

    gamma = 0.99
    learning_rate = 0.001
    epsilon_current = 0.1
    epsilon_decrease = 1
    epsilon_min = 0.05
    episode_gains = []

    max_episodes = 100000

    solver = Solver(
        # learning parameters
        learning_rate=learning_rate,
        # feature extraction parameters
        number_of_kernels_per_dim=[7, 5],
        # env dependencies (DO NOT CHANGE):
        number_of_actions=env.action_space.n,
    )

    for episode_index in range(1, max_episodes + 1):
    # for episode_index in range(1, 2):
        episode_gain = run_episode(env, solver, alpha=1, is_train=True, epsilon=epsilon_current)

        # reduce epsilon if required
        epsilon_current *= epsilon_decrease
        epsilon_current = max(epsilon_current, epsilon_min)

        print(f'after {episode_index}, reward = {episode_gain}, epsilon {epsilon_current}')

        # termination condition:
        if episode_index % 10 == 9:
            test_gains = [run_episode(env, solver, alpha=1, is_train=False, epsilon=0.) for _ in range(10)]
            print(test_gains)
            mean_test_gain = np.mean(test_gains)
            episode_gains.append(mean_test_gain)
            print(f'tested 10 episodes: mean gain is {mean_test_gain}')
            if mean_test_gain >= -75.:
                print(f'solved in {episode_index} episodes')
                break

    run_episode(env, solver, alpha=1, is_train=False, render=True)
    # Create a line plot
    plt.plot(episode_gains)

    # Label the axes
    plt.xlabel("Index")
    plt.ylabel("Mean Rewards")

    # Set the title of the plot
    plt.title("Mean Rewards Plot")

    # Show the plot
    plt.show()