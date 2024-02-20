import numpy as np


class DataCollector:
    def __init__(self, env_with_reset):
        self._env_with_reset = env_with_reset

    def state_selection(self):
        reticle_x = np.random.randint(self._env_with_reset.min_position_x, self._env_with_reset.max_position_x)
        target_x = np.random.randint(self._env_with_reset.min_position_x, self._env_with_reset.max_position_x)
        reticle_y = np.random.randint(self._env_with_reset.min_position_y, self._env_with_reset.max_position_y)
        target_y = np.random.randint(self._env_with_reset.min_position_y, self._env_with_reset.max_position_y)
        return (reticle_x, reticle_y), (target_x, target_y)

    def action_selection(self):
        return np.random.choice(4)

    def collect_data(self, number_of_samples):
        # result should be (s_t, a_t, r_t, s_{t+1})
        result = []
        for _ in range(number_of_samples):
            state = self.state_selection()
            state = self._env_with_reset.reset_specific(state[0], state[1])
            action = self.action_selection()
            next_state, reward, done, _ = self._env_with_reset.step(action)
            result_tuple = (state, action, reward, next_state, done)
            result.append(result_tuple)
        return self.process_data(result)

    @staticmethod
    def process_data(data):
        states, actions, rewards, next_states, done_flags = zip(*data)
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_states = np.array(next_states)
        done_flags = np.array(done_flags)
        return states, actions, rewards, next_states, done_flags
