import numpy as np

from mountain_car_with_data_collection import MountainCarWithResetEnv
from data_collector import DataCollector
from data_transformer import DataTransformer
from radial_basis_function_extractor import RadialBasisFunctionExtractor
from linear_policy import LinearPolicy
from game_player import GamePlayer

class Policy:
    def __init__(self, env, data_transformer, feature_extractor, W, epsilon):
        self.env = env
        self.data_transformer = data_transformer
        self.feature_extractor = feature_extractor
        self.number_of_actions = 3
        self.W = W
        self.epsilon = epsilon

    def modify_reward(self, reward):
        reward -= 1
        if reward == 0.:
            reward = 100.
        return reward

    def policy_gradient_iteration(self, feature_extractor, alpha, tau=100, epsilon=0.001):
        """
        Performs one iteration of policy gradient algorithm.
        :param feature_extractor: Feature extractor object
        :param W: Current weights
        :param alpha: Learning rate
        :param tau: Number of steps to perform in each rollout
        :param epsilon: Update step size
        :return: New Weight
        """

        rewards_sum = 0
        gradients_sum = 0
        states_array = []
        x0 = self.env.reset()
        encoded_state = feature_extractor.encode_states_with_radial_basis_functions(np.array([x0]))
        #perfroming a rollout
        for i in range (tau):
            _, action = self.softmax_policy(feature_extractor, encoded_state, alpha)
            state, reward, done, _ = env.step(action)
            states_array.append(state)
            encoded_state = feature_extractor.encode_states_with_radial_basis_functions(np.array([state]))
            # encoded_state = evaluator._process_single_state(states)
            # rewards.append(reward)
            # gradients.append(self.log_softmax_gradient(feature_extractor, alpha, encoded_state, action))
            rewards_sum += self.modify_reward(reward)
            gradients_sum += self.log_softmax_gradient(feature_extractor, alpha, encoded_state, action)
            if done:
                break
        
        #computing the gradient
        grad = rewards_sum * gradients_sum
        #updating the weights
        self.W = self.W + epsilon * grad.T
        return self.W, states_array

    def softmax_policy(self, feature_extracor, feature_vector, alpha):
        array = []
        sum = 0 
        for i in range(3):
            encoded_states_tiled = feature_extractor.tiling(feature_vector, i)
            up = np.exp(alpha*encoded_states_tiled@self.W - max(alpha*encoded_states_tiled@self.W))
            array.append(up)
            sum += up

        policy = array / sum
        if np.random.rand() < self.epsilon:
            action = np.random.choice(len(policy))
        else:
            action = np.argmax(policy)
        return policy, action

    def log_softmax_gradient(self, feature_extractor, alpha, encoded_states, action):
        up_sum = np.zeros(len(encoded_states[0]) * 3) 
        down_sum = 0 # dim is 1 (exp of inner prod) 
        for i in range(3):
            # print('shape before tile:', encoded_states.shape)
            encoded_states_tiled = feature_extractor.tiling(encoded_states, i)
            # print('shape after tile:', encoded_states_tiled.shape)
            A = alpha*encoded_states_tiled.T @ np.exp(alpha*encoded_states_tiled@self.W - max(encoded_states_tiled@self.W))
            # print(A.reshape(-1).shape)
            A=A.reshape(-1)
            up_sum += A
            down_sum += np.exp(alpha*encoded_states_tiled@self.W - max(alpha*encoded_states_tiled@self.W))
        b = up_sum / down_sum
        a = alpha * feature_extractor.tiling(encoded_states, action)
        return a - b
    
    def get_max_action(self, encoded_state):
        _, action = self.softmax_policy(self.feature_extractor, encoded_state, alpha=1)
        return action




# def log_softmax_gradient(W, feature_vector, alpha):
#     y = softmax_policy(W, feature_vector, alpha)
#     z = feature_vector @ W
#     z_max = np.max(z)  # Numerical stability trick
#     y_tilde = np.exp(z - z_max) / np.sum(np.exp(z - z_max))
#     return feature_vector*(y - y_tilde)

if __name__ == '__main__':
    samples_to_collect = 100000
    # samples_to_collect = 150000
    # samples_to_collect = 10000
    number_of_kernels_per_dim = [12, 10]
    gamma = 0.99
    w_updates = 100
    evaluation_number_of_games = 10
    evaluation_max_steps_per_game = 1000

    np.random.seed(123)
    # np.random.seed(234)

    env = MountainCarWithResetEnv()
    # # collect data
    # states, actions, rewards, next_states, done_flags = DataCollector(env).collect_data(samples_to_collect)
    # # get data success rate
    # data_success_rate = np.sum(rewards) / len(rewards)
    # print(f'success rate {data_success_rate}')
    # # standardize data
    data_transformer = DataTransformer()
    # data_transformer.set_using_states(np.concatenate((states, next_states), axis=0))
    # states = data_transformer.transform_states(states)
    # next_states = data_transformer.transform_states(next_states)
    # # process with radial basis functions
    feature_extractor = RadialBasisFunctionExtractor(number_of_kernels_per_dim)
    # encode all states:
    # encoded_states = feature_extractor.encode_states_with_radial_basis_functions(states)    
    # print(encoded_states.shape)
    # encoded_next_states = feature_extractor.encode_states_with_radial_basis_functions(next_states)
    # # set a new linear policy
    linear_policy = LinearPolicy(feature_extractor.get_number_of_features(), 3, False)
    # but set the weights as random
    linear_policy.set_w(np.random.uniform(size=linear_policy.w.shape))
    W = linear_policy.w
    print(W)
    # policy , action = softmax_policy(linear_policy.w, feature_extractor, encoded_states[1, :], alpha=1)
    # print(action)
    # grad = log_softmax_gradient(linear_policy.w, feature_extractor, alpha=1, encoded_states=encoded_states[1,:], action=0)
    # print(grad)
    # # start an object that evaluates the success rate over time
    policy = Policy(env, data_transformer, feature_extractor,  W, epsilon=0.3)
    # state = env.reset()
    # evaluator.play_game(evaluation_max_steps_per_game, start_state=state)
    success_rate = []
    for pol_grad_iteration in range(w_updates):
        print(f'starting policy gradient iteration {pol_grad_iteration}')

        new_W, states = policy.policy_gradient_iteration(feature_extractor, alpha=1, tau=evaluation_max_steps_per_game, epsilon=0.001)
        # linear_policy.set_w(new_W)
        # new_w = policy_gradient_iteration(
        #     encoded_states, encoded_next_states, actions, rewards, done_flags, linear_policy, gamma
        # )
        norm_diff = linear_policy.set_w(new_W)
        if norm_diff < 0.00001:
            break
        data_transformer.set_using_states(states)
        states = data_transformer.transform_states(states)
        policy = Policy(env, data_transformer, feature_extractor,  new_W, epsilon=0.3)
        evaluator = GamePlayer(env, data_transformer, feature_extractor, policy)
        success_rate.append(evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game))
    print('done policy gradient')
    evaluator.play_games(evaluation_number_of_games, evaluation_max_steps_per_game)
    evaluator.play_game(evaluation_max_steps_per_game, render=True)
    
