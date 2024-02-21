import numpy as np


class Policy:
    def __init__(self, env, feature_extractor, W, epsilon):
        self.env = env
        self.feature_extractor = feature_extractor
        self.number_of_actions = 3
        self.W = W
        self.epsilon = epsilon

    def policy_gradient_iteration(self, alpha, tau=100, epsilon=0.001):
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
        state = self.env.reset()
        #perfroming a rollout
        for i in range (tau):
            _, action = self.softmax_policy(state, alpha)
            state, reward, done, _ = self.env.step(action)
            states_array.append(state)
            # encoded_state = evaluator._process_single_state(states)
            # rewards.append(reward)
            # gradients.append(self.log_softmax_gradient(feature_extractor, alpha, encoded_state, action))
            rewards_sum += reward
            gradients_sum += self.log_softmax_gradient(alpha, state, action)
            if done:
                break
        
        #computing the gradient
        grad = rewards_sum * gradients_sum
        #updating the weights
        self.W = self.W + epsilon * grad.T
        return self.W, states_array

    def softmax_policy(self, encoded_state, alpha, epsilon=0.001):
        array = []
        sum = 0 
        for i in range(3):
            encoded_states_tiled = self.feature_extractor.tiling(encoded_state, i)
            up = np.exp(alpha*encoded_states_tiled@self.W - max(alpha*encoded_states_tiled@self.W))
            array.append(up)
            sum += up

        policy = array / sum
        if np.random.rand() < epsilon:
            action = np.random.choice(len(policy))
        else:
            action = np.argmax(policy)
        return policy, action

    def log_softmax_gradient(self, alpha, encoded_states, action):
        up_sum = np.zeros(len(encoded_states[0]) * 5)
        down_sum = 0 # dim is 1 (exp of inner prod) 
        for i in range(3):
            # print('shape before tile:', encoded_states.shape)
            encoded_states_tiled = self.feature_extractor.tiling(encoded_states, i)
            # print('shape after tile:', encoded_states_tiled.shape)
            up_value = alpha*encoded_states_tiled.T @ np.exp(alpha*encoded_states_tiled@self.W - max(encoded_states_tiled@self.W))
            # print(A.reshape(-1).shape)
            up_value.reshape(-1)
            up_sum += up_value
            down_sum += np.exp(alpha*encoded_states_tiled@self.W - max(alpha*encoded_states_tiled@self.W))
        b = up_sum / down_sum
        a = alpha * self.feature_extractor.tiling(encoded_states, action)
        return a - b
    
    def get_max_action(self, encoded_state):
        _, action = self.softmax_policy(encoded_state, alpha=1)
        return action

    def set_w(self, w):
        assert self.w.shape == w.shape
        change = np.linalg.norm(w - self.w)
        print(f'changed w, norm diff is {change}')
        self.w = w
        return change




# def log_softmax_gradient(W, feature_vector, alpha):
#     y = softmax_policy(W, feature_vector, alpha)
#     z = feature_vector @ W
#     z_max = np.max(z)  # Numerical stability trick
#     y_tilde = np.exp(z - z_max) / np.sum(np.exp(z - z_max))
#     return feature_vector*(y - y_tilde)


    
