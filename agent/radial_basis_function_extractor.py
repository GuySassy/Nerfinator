import numpy as np
import itertools


class RadialBasisFunctionExtractor:
    def __init__(self, number_of_kernels_per_dim):
        lower = -2.
        upper = 2.
        mus_per_dim = []
        for number_of_kernels in number_of_kernels_per_dim:
            direction = (upper - lower) / (number_of_kernels - 1)
            mus = [lower + i * direction for i in range(number_of_kernels)]
            mus_per_dim.append(mus)
        self.mus = [np.array(mu) for mu in itertools.product(*mus_per_dim)]
        self.sigmas = [2. / (min(number_of_kernels_per_dim) - 1)] * len(self.mus)

    @staticmethod
    def _compute_kernel(states, mu, sigma):
        exponent = np.linalg.norm(states - mu, axis=1)
        exponent = -exponent / (2 * np.square(sigma))
        return np.exp(exponent)

    def encode_states_with_radial_basis_functions(self, states):
        features = [
            np.expand_dims(self._compute_kernel(states, self.mus[i], self.sigmas[i]), axis=1)
            for i in range(self.get_number_of_features())
        ]
        return np.concatenate(features, axis=1)

    def get_number_of_features(self):
        return len(self.mus)
    
    def tiling(self, feature_vector, action):
        assert action in [0, 1, 2]
        tiled_feature_vector = np.zeros((1, self.get_number_of_features() * 3))
        tiled_feature_vector[0, action * self.get_number_of_features():(action + 1) * self.get_number_of_features()] = feature_vector
        return tiled_feature_vector
    
    
if __name__ == '__main__':
    extractor = RadialBasisFunctionExtractor([12, 10])
    states = np.array([[0.0, 0.0]])
    features = extractor.encode_states_with_radial_basis_functions(states)
    tiled_features = extractor.tiling(features, 1)
    print("The features are:", features)
    print("The shape is:", features.shape)
    print("The tiled features are:", tiled_features)
    print("The shape is:", tiled_features.shape)
    print(extractor.get_number_of_features())


        

