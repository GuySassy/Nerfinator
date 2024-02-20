import numpy as np


class DataTransformer:
    def __init__(self):
        self.state_mean = None
        self.state_std = None

    def set_using_states(self, all_states):
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0)
        self.set(state_mean, state_std)

    def set(self, state_mean, state_std):
        self.state_mean = state_mean
        self.state_std = state_std
        print(f'data mean {state_mean}')
        print(f'data std {state_std}')

    def transform_states(self, states):
        standardized_states = states - self.state_mean
        standardized_states = standardized_states / self.state_std
        return standardized_states
    
if __name__ == '__main__':
    data_transformer = DataTransformer()
    state = np.array([0.1, 0.2])
    print(state.shape)
    state = np.expand_dims(state, axis=0)
    print(state.shape)
    state = data_transformer.transform_states(state)