import numpy as np

class InfiniteHiddenMarkovModel():
    """
    Infinite Hidden Markov Model
    """
    def __init__(self,alpha,beta,gamma,beta0,gamma0):
        # Hyperparameters
        self.alpha = alpha # Controls tendency to linger in the same state
        self.beta = beta # Influences the tendency to explore new states
        self.gamma = gamma  # Controls the tendency to create new states
        self.beta0 = beta0 # Influences the tendency to explore new observations 
        self.gamma0 = gamma0 # Controls the tendency to create new observations

        # Transitions
        self.transition_counts = {} # counts transition (nij)
        self.oracle_counts = {} # counts state (noj)
        self.num_states = 0 # number of states (clusters)

        # Emmisions
        self.emission_counts = {} # counts emission (njk)
        self.num_emissions = 0 # number of observation

        self.state_sequence = []


    def initialize_state(self):
        """
        Initialize state
        """
        self.num_states += 1
        state_id = self.num_states
        self.transition_counts[state_id] = {}
        self.emission_counts[state_id] = {}
        self.oracle_counts[state_id] = 0
        return state_id
    
    def sample_next_state(self, current_state):
        """
        Sample next state
        Test: Cleared (All probabilities sum to 1)
        """
        # Ensure that the current state exists
        if current_state is None or current_state not in self.transition_counts:
            new_state = self.initialize_state()
            self.oracle_counts[new_state] = 1
            return new_state

        # Calculating the transition probabilities
        total_transitions = sum(self.transition_counts[current_state].values())
        denominator = total_transitions + self.alpha + self.beta

        probs = {}
        # Probabilities of transitions from current_state to other states
        for state, count in self.transition_counts[current_state].items():
            probs[state] = count / denominator

        # Probability of creating a new state
        prob_new = self.beta / denominator

        # Probability of staying in the same state
        probs[current_state] = (
            self.alpha + self.transition_counts[current_state].get(current_state, 0)
        ) / denominator

        # Normalizing the probabilities
        total_prob = sum(probs.values()) + prob_new
        for state in probs:
            probs[state] /= total_prob
        prob_new /= total_prob

        # Sample next state
        if np.random.rand() < prob_new:
            next_state = self.consult_oracle()
        else:
            next_state = np.random.choice(list(probs.keys()), p=list(probs.values()))

        # Update counts
        self.update_counts(current_state, next_state)
        return next_state
        
    def consult_oracle(self):
        """
        Consult the oracle
        """
        total_oracle = sum(self.oracle_counts.values()) # Count of noj
        prob_new = self.gamma / (total_oracle + self.gamma) # Probability of creating a new state
        probs = {}
        for state,count in self.oracle_counts.items(): 
            probs[state] = count / (total_oracle + self.gamma)

        if np.random.rand() < prob_new:
            # Create a new state with probability gamma
            new_state =  self.initialize_state()

        else:
            # Sample from the oracle
            new_state = np.random.choice(list(probs.keys()),p=list(probs.values()))

        self.oracle_counts[new_state] += 1
        return new_state
    
    def update_counts(self,current_state,next_state):
        """
        Update counts
        """
        if next_state not in self.transition_counts[current_state]:
            self.transition_counts[current_state][next_state] = 0

        self.transition_counts[current_state][next_state] += 1

    def sample_emission(self,state):
        """
        Sample emission
        """
        # Ensure that the current observation exists
        if state not in self.emission_counts:
            self.emission_counts[state] = {}
        
        # Calculating the emission probabilities
        total_emissions = sum(self.emission_counts[state].values())

        probs = {}
        # Probabilities of j to k
        for emission,count in self.emission_counts[state].items():
            probs[emission] = count / (total_emissions + self.beta0)
        
        # Probability of creating a new observation
        prob_new = self.beta0 / (total_emissions + self.beta0)

        # Sample next observation
        if np.random.rand() < prob_new:
            next_emission = self.create_new_emission()
        
        else:
            next_emission = np.random.choice(list(probs.keys()),p=list(probs.values()))

        # Update emmision counts
        self.update_emission_counts(state,next_emission)
        return next_emission
    
    def create_new_emission(self):
        """
        Create new emission
        """
        self.num_emissions += 1
        return f"obs_{self.num_emissions}"
    
    def update_emission_counts(self, state, emission):
        """
        Update emission counts for a given state
        """
        if emission not in self.emission_counts[state]:
            self.emission_counts[state][emission] = 0
        self.emission_counts[state][emission] += 1

    def gibbs_sampling(self, observations, num_iter=1000):
        """
        Gibbs sampling for hidden state sequence
        """
        if not observations:
            raise ValueError("Observations sequence cannot be empty")
        
        T = len(observations)
        self.state_sequence = [self.initialize_state() for _ in range(T)]

        for iteration in range(num_iter):
            for t in range(T):
                current_state = self.state_sequence[t]
                prev_state = self.state_sequence[t - 1] if t > 0 else None
                next_state = self.state_sequence[t + 1] if t < T - 1 else None

                # Remove current state's influence on counts
                if prev_state:
                    self.transition_counts[prev_state][current_state] -= 1
                    if self.transition_counts[prev_state][current_state] == 0:
                        del self.transition_counts[prev_state][current_state]

                if current_state in self.emission_counts:
                    self.emission_counts[current_state][observations[t]] -= 1
                    if self.emission_counts[current_state][observations[t]] == 0:
                        del self.emission_counts[current_state][observations[t]]

                # Sample a new state
                self.state_sequence[t] = self.sample_next_state(prev_state)

                # Update transition and emission counts
                if prev_state:
                    self.update_counts(prev_state, self.state_sequence[t])
                self.update_emission_counts(self.state_sequence[t], observations[t])

    def particle_filtering(self, observations, num_particles=100):
        """
        Perform infinite-state particle filtering for likelihood computation
        """
        T = len(observations)
        particles = [self.initialize_state() for _ in range(num_particles)]
        log_likelihood = 0

        for t in range(T):
            weights = []
            for particle in particles:
                if observations[t] in self.emission_counts.get(particle, {}):
                    prob = self.emission_counts[particle][observations[t]] / sum(self.emission_counts[particle].values())
                else:
                    prob = self.beta0 / (self.beta0 + sum(self.emission_counts.get(particle, {}).values()))
                weights.append(prob)

            weights = np.array(weights)
            weights /= weights.sum()

            log_likelihood += np.log(weights.sum())

            resampled_particles = np.random.choice(particles, size=num_particles, p=weights)
            particles = []

            for particle in resampled_particles:
                next_state = self.sample_next_state(particle)
                particles.append(next_state)

        return log_likelihood

    def predict(self, initial_state, steps):
        """
        Predict a sequence of observations starting from an initial state
        """
        if steps <= 0:
            raise ValueError("Number of steps must be positive")
        

        state = initial_state
        predictions = []

        for _ in range(steps):
            observation = self.sample_emission(state)
            predictions.append(observation)
            state = self.sample_next_state(state)

        return predictions
    
def main():
    try:
        print("Initializing IHMM...")
        ihmm = InfiniteHiddenMarkovModel(alpha=1.0, beta=2.0, gamma=1.0, beta0=1.0, gamma0=1.0)

        observations = ["obs_1", "obs_2", "obs_3", "obs_2", "obs_1","obs_2","obs_3","obs_2", "obs_1","obs_2","obs_3"]
        print(f"\nObservations: {observations}")

        print("\nPerforming Gibbs sampling...")
        ihmm.gibbs_sampling(observations, num_iter=1000)
        print("Hidden state sequence:", ihmm.state_sequence)

        print("\nPredicting future observations...")
        initial_state = ihmm.state_sequence[-1]
        predictions = ihmm.predict(initial_state, steps=1)
        print("Predicted observations:", predictions)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()