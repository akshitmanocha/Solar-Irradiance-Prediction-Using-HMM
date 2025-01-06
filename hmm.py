import numpy as np

class HiddenMarkovModel():
    """"
    Stores a hidden markov object, and model parameters
    
    Algorithms Implemented:
    1) Viterbi Algorithm
    2) Forward Algorithm
    3) Backward Algorithm
    4) Baum-Welch Algorithm

    Args:
        states: list of states
        observations: list of observations
        start_probability: stationary probability matrix (N x 1) 
        transition_probability: transition probability matrix (N x N)
        emission_probability: emission probability matrix (N x M)    
    """

    def __init__(self, states, observations, start_probability, transition_probability, emission_probability):
        self.states = np.asarray(states)
        self.observations = np.asarray(observations)
        self.N = len(self.states)
        self.M = len(self.observations)
        self.start_probability = start_probability
        self.transition_probability = transition_probability
        self.emission_probability = emission_probability
        self.obs_map = {obs: idx for idx, obs in enumerate(self.observations)}

    def _get_obs_idx(self, obs):
        """Convert observation to index"""
        return self.obs_map[obs]

    def ForwardAlgorithm(self, obs_seq):
        """
        Forward Algorithm
        Args:
            obs_seq: list of observations
        Returns:
            likelihood: likelihood of the observation sequence
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        obs_idx = self._get_obs_idx(obs_seq[0])
        alpha[0] = self.start_probability * self.emission_probability[:, obs_idx]
        
        for t in range(1, T):
            obs_idx = self._get_obs_idx(obs_seq[t])
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probability[:, j]) * self.emission_probability[j, obs_idx]

        return np.sum(alpha[T-1])

    def BackwardAlgorithm(self, obs_seq):
        """
        Backward Algorithm
        Args:
            obs_seq: list of observations
        Returns:
            likelihood: likelihood of the observation sequence
        """
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1
        
        for t in range(T-2, -1, -1):
            obs_idx = self._get_obs_idx(obs_seq[t+1])
            for i in range(self.N):
                beta[t, i] = np.sum(self.transition_probability[i, :] * self.emission_probability[:, obs_idx] * beta[t+1])

        obs_idx = self._get_obs_idx(obs_seq[0])
        return np.sum(self.start_probability * self.emission_probability[:, obs_idx] * beta[0])

    def return_alpha(self, obs_seq):
        """
        Returns the alpha matrix
        Args:
            obs_seq: list of observations
        Returns:
            alpha: alpha matrix
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        obs_idx = self._get_obs_idx(obs_seq[0])
        alpha[0] = self.start_probability * self.emission_probability[:, obs_idx]
        
        for t in range(1, T):
            obs_idx = self._get_obs_idx(obs_seq[t])
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t-1] * self.transition_probability[:, j]) * self.emission_probability[j, obs_idx]
        return alpha

    def return_beta(self, obs_seq):
        """
        Returns the beta matrix
        Args:
            obs_seq: list of observations
        Returns:
            beta: beta matrix
        """
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[T-1] = 1
        
        for t in range(T-2, -1, -1):
            obs_idx = self._get_obs_idx(obs_seq[t+1])
            for i in range(self.N):
                beta[t, i] = np.sum(self.transition_probability[i, :] * self.emission_probability[:, obs_idx] * beta[t+1])
        return beta

    def return_gamma(self, obs_seq):
        """
        Returns the gamma matrix
        Args:
            obs_seq: list of observations
        Returns:
            gamma: gamma matrix
        """
        alpha = self.return_alpha(obs_seq)
        beta = self.return_beta(obs_seq)
        gamma = alpha * beta
        gamma = gamma / (np.sum(gamma, axis=1, keepdims=True) + 1e-12)
        return gamma

    def ViterbiAlgorithm(self, obs_seq):
        """
        Viterbi Algorithm
        Args:
            obs_seq: list of observations
        Returns:
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        obs_idx = self._get_obs_idx(obs_seq[0])
        delta[0] = self.start_probability * self.emission_probability[:, obs_idx]
        psi[0] = 0

        for t in range(1, T):
            obs_idx = self._get_obs_idx(obs_seq[t])
            for j in range(self.N):
                delta[t, j] = np.max(delta[t-1] * self.transition_probability[:, j]) * self.emission_probability[j, obs_idx]
                psi[t, j] = np.argmax(delta[t-1] * self.transition_probability[:, j])

        return np.max(delta[T-1]), np.argmax(delta[T-1])

    def BaumWelchAlgorithm(self, obs_seq, iterations):
        """
        Baum-Welch Algorithm
        Args:
            obs_seq: list of observations
            iterations: number of iterations
        Returns:
            start_probability: updated start probability matrix
            transition_probability: updated transition probability matrix
            emission_probability: updated emission probability matrix
        """
        T = len(obs_seq)
        obs_seq_idx = [self._get_obs_idx(obs) for obs in obs_seq]
        
        for _ in range(iterations):
            # E-Step
            alpha = self.return_alpha(obs_seq)
            beta = self.return_beta(obs_seq)
            gamma = np.zeros((T, self.N))
            xi = np.zeros((T-1, self.N, self.N))
            
            # Compute gamma and xi
            for t in range(T):
                gamma[t] = alpha[t] * beta[t] / (np.sum(alpha[t] * beta[t]) + 1e-12)
                if t < T-1:
                    obs_idx = self._get_obs_idx(obs_seq[t+1])
                    for i in range(self.N):
                        for j in range(self.N):
                            xi[t, i, j] = alpha[t, i] * self.transition_probability[i, j] * \
                                        self.emission_probability[j, obs_idx] * beta[t+1, j]
                    xi[t] /= (np.sum(xi[t]) + 1e-12)
            
            # M-Step
            self.start_probability = gamma[0]
            for i in range(self.N):
                self.transition_probability[i] = np.sum(xi[:, i, :], axis=0) / (np.sum(xi[:, i, :]) + 1e-12)
                for k in range(self.M):
                    mask = np.array(obs_seq_idx) == k
                    self.emission_probability[i, k] = np.sum(gamma[mask, i]) / (np.sum(gamma[:, i]) + 1e-12)
        
        return self.start_probability, self.transition_probability, self.emission_probability

    def predict(self, obs_seq):
        """
        Predicts the most likely state sequence given the observation sequence
        Args:
            obs_seq: list of observations
        Returns:
            state_sequence: most likely state sequence
        """
        T = len(obs_seq)
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)

        obs_idx = self._get_obs_idx(obs_seq[0])
        delta[0] = self.start_probability * self.emission_probability[:, obs_idx]
        psi[0] = 0

        for t in range(1, T):
            obs_idx = self._get_obs_idx(obs_seq[t])
            for j in range(self.N):
                delta[t, j] = np.max(delta[t-1] * self.transition_probability[:, j]) * self.emission_probability[j, obs_idx]
                psi[t, j] = np.argmax(delta[t-1] * self.transition_probability[:, j])

        state_sequence = []
        q_star = np.argmax(delta[T-1])
        state_sequence.append(q_star)

        for t in range(T-1, 0, -1):
            q_star = psi[t, q_star]
            state_sequence.append(q_star)

        return list(reversed(state_sequence))
    


if __name__ == "__main__":
    states = np.array(['rainy', 'cloudy', 'happy']) 
    observations = np.array(['happy', 'sad', 'meh'])

    A = np.array([
        [0.6, 0.2, 0.2],  # rainy transitions
        [0.5, 0.3, 0.2],  # cloudy transitions
        [0.4, 0.1, 0.5]   # happy transitions
    ])
    pi = np.array([0.5, 0.2, 0.3])
    B = np.array([
        [0.3, 0.4, 0.3],  # rainy emissions
        [0.2, 0.4, 0.4],  # cloudy emissions
        [0.5, 0.1, 0.4]   # happy emissions
    ])

    hmm = HiddenMarkovModel(states, observations, pi, A, B)
    obs_sequence = np.array(['sad', 'meh', 'happy'])

    # Forward Algorithm
    forward_likelihood = hmm.ForwardAlgorithm(obs_sequence)
    print(f"Forward Likelihood: {forward_likelihood}")

    # Backward Algorithm
    backward_likelihood = hmm.BackwardAlgorithm(obs_sequence)
    print(f"Backward Likelihood: {backward_likelihood}")

    # Viterbi Algorithm
    best_prob, best_state = hmm.ViterbiAlgorithm(obs_sequence)
    print(f"Most likely state probability: {best_prob}")
    print(f"Final state: {states[best_state]}")  # Convert state index to label

    # Get full state sequence prediction
    predicted_states = hmm.predict(obs_sequence)
    print(f"Predicted state sequence: {[states[i] for i in predicted_states]}")  # Convert indices to labels

    # Get state probabilities at each time step
    gamma = hmm.return_gamma(obs_sequence)
    print(f"State probabilities at each time step:")
    for t in range(len(obs_sequence)):
        print(f"Time {t} ({obs_sequence[t]}):")
        for s in range(len(states)):
            print(f"  {states[s]}: {gamma[t,s]:.3f}")