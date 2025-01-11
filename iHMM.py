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
        self.emission_oracle_counts = {} # counts emission (noj)
        self.num_emissions = 0 # number of observation

        self.state_sequence = [] # Hidden state sequence


    def initialize_state(self):
        """
        Initialize state
        """
        try:
            self.num_states += 1
            state_id = self.num_states
            self.transition_counts[state_id] = {}
            self.emission_counts[state_id] = {}
            self.oracle_counts[state_id] = 0
            return state_id
        
        except Exception as e:
            raise Exception(f"Error in initialize_state: {str(e)}")
    
    def sample_next_state(self, current_state):
        """
        Sample next state.
        Test: Cleared (All probabilities sum to 1).
        """
        # Ensure that the current state exists
        if current_state is None or current_state not in self.transition_counts:
            new_state = self.initialize_state()
            self.oracle_counts[new_state] = 1
            return new_state

        # Calculating the transition probabilities
        total_transitions = sum(self.transition_counts[current_state].values())

        # Calculating the denominator
        denominator = total_transitions + self.alpha + self.beta
        if denominator == 0:
            raise ValueError(f"Zero denominator in sample_next_state for state {current_state}")

        # Calculate probabilities for existing states
        probs = {
            state: (count + (self.alpha if state == current_state else 0)) / denominator
            for state, count in self.transition_counts[current_state].items()
        }

        # Add oracle probability
        prob_oracle = self.beta / denominator
        probs["oracle"] = prob_oracle

        # Normalize probabilities to sum to 1
        keys = list(probs.keys())
        probabilities = np.array([probs[k] for k in keys])
        probabilities /= probabilities.sum()

        # Sample next state
        sampled_key = np.random.choice(keys, p=probabilities)
        if sampled_key == "oracle":
            next_state = self.consult_oracle()
        else:
            next_state = sampled_key

        # Update counts
        self.update_counts(current_state, next_state)
        return next_state
        
    def consult_oracle(self):
        """
        Consult the oracle.
        """
        total_oracle = sum(self.oracle_counts.values())  # Total count of oracle states
        prob_new = self.gamma / (total_oracle + self.gamma)  # Probability of creating a new state

        # Handle edge case where oracle counts are empty
        if total_oracle == 0:
            prob_new = 1.0

        # Calculate probabilities for existing states
        probs = {
            state: count / (total_oracle + self.gamma)
            for state, count in self.oracle_counts.items()
        }

        # Normalize probabilities to sum to 1
        probs["new_state"] = prob_new
        keys = list(probs.keys())
        probabilities = np.array([probs[key] for key in keys])
        probabilities /= probabilities.sum()

        # Sample new state or an existing state
        sampled_key = np.random.choice(keys, p=probabilities)
        if sampled_key == "new_state":
            new_state = self.initialize_state()
            self.oracle_counts[new_state] = 1  # Initialize count for the new state
        else:
            new_state = sampled_key

        # Increment count for the chosen state
        self.oracle_counts[new_state] += 1
        return new_state
    
    def update_counts(self, current_state, next_state):
        """
        Update counts for state transitions.
        """
        # Ensure the current_state exists in transition_counts
        if current_state not in self.transition_counts:
            self.transition_counts[current_state] = {}

        # Ensure the next_state exists in the dictionary for current_state
        if next_state not in self.transition_counts[current_state]:
            self.transition_counts[current_state][next_state] = 0

        # Increment the transition count
        self.transition_counts[current_state][next_state] += 1

    def sample_emission(self, state):
        """
        Sample emission for a given state.
        """
        # Ensure the state exists in emission_counts
        if state not in self.emission_counts:
            self.emission_counts[state] = {}

        # Calculate total emissions for the state
        total_emissions = sum(self.emission_counts[state].values())

        # Handle case where no emissions exist
        if total_emissions == 0:
            prob_new = 1.0  # All probability goes to creating a new observation
        else:
            prob_new = self.beta0 / (total_emissions + self.beta0)

        # Calculate probabilities for existing emissions
        probs = {
            emission: count / (total_emissions + self.beta0)
            for emission, count in self.emission_counts[state].items()
        }

        # Add the probability for creating a new observation
        probs["new_emission"] = prob_new

        # Normalize probabilities to sum to 1
        keys = list(probs.keys())
        probabilities = np.array([probs[key] for key in keys])
        probabilities /= probabilities.sum()

        # Sample next emission
        sampled_key = np.random.choice(keys, p=probabilities)
        if sampled_key == "new_emission":
            next_emission = self.create_new_emission()
        else:
            next_emission = sampled_key

        # Update emission counts
        self.update_emission_counts(state, next_emission)
        return next_emission
    
    def create_new_emission(self):
        """
        Create a new unique emission.
        """
        if not hasattr(self, 'num_emissions'):
            self.num_emissions = 0

        self.num_emissions += 1
        return f"obs_{self.num_emissions}"
    
    def update_emission_counts(self, state, emission):
        """
        Update emission counts for a given state.
        """
        # Ensure the state exists in emission_counts
        if state not in self.emission_counts:
            self.emission_counts[state] = {}

        # Ensure the emission exists in the dictionary for the state
        if emission not in self.emission_counts[state]:
            self.emission_counts[state][emission] = 0

        # Increment the emission count
        self.emission_counts[state][emission] += 1

    def gibbs_sampling(self, observations, num_iter=1000):
        """
        Perform Gibbs sampling to infer the hidden state sequence in an iHMM.
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
                if current_state in self.transition_counts and next_state:
                    self.transition_counts[current_state][next_state] -= 1
                    if self.transition_counts[current_state][next_state] == 0:
                        del self.transition_counts[current_state][next_state]

                # Sample a new state
                self.state_sequence[t] = self.sample_next_state(prev_state)

                # Update transition counts for the new state
                if prev_state:
                    self.update_counts(prev_state, self.state_sequence[t])
                if next_state:
                    self.update_counts(self.state_sequence[t], next_state)
                
                # Update emission counts for the new state
                self.update_emission_counts(self.state_sequence[t], observations[t])

            # Optional: Hyperparameter optimization
            self.optimize_hyperparameters()

        return self.state_sequence

    def particle_filtering(self, observations, num_particles=100):
        """
        Perform infinite-state particle filtering for likelihood computation.
        """
        T = len(observations)
        if T == 0:
            raise ValueError("Observations cannot be empty")

        # Initialize particles
        particles = [self.initialize_state() for _ in range(num_particles)]
        log_likelihood = 0

        for t in range(T):
            weights = []

            # Calculate weights for each particle
            for particle in particles:
                total_emissions = sum(self.emission_counts.get(particle, {}).values())
                if observations[t] in self.emission_counts.get(particle, {}):
                    prob = self.emission_counts[particle][observations[t]] / (total_emissions + self.beta0)
                else:
                    prob = self.beta0 / (total_emissions + self.beta0)
                weights.append(prob)

            weights = np.array(weights)

            # Handle edge case: all weights are zero
            if weights.sum() == 0:
                weights = np.ones_like(weights) / len(weights)
            else:
                weights /= weights.sum()  # Normalize weights

            # Update log likelihood
            log_likelihood += np.log(weights.sum() + 1e-10)

            # Resample particles based on weights
            resampled_particles = np.random.choice(particles, size=num_particles, p=weights, replace=True)

            # Transition to next states
            particles = [self.sample_next_state(particle) for particle in resampled_particles]

        return log_likelihood

    def predict(self, initial_state, steps):
        """
        Predict a sequence of observations starting from an initial state.

        Args:
            initial_state: The state to start the prediction from.
            steps: The number of observations to predict.

        Returns:
            A list of predicted observations.
        """
        if steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        if initial_state not in self.transition_counts:
            raise ValueError(f"Initial state '{initial_state}' not found in the state space")

        state = initial_state
        predictions = []

        for _ in range(steps):
            # Sample emission from the current state
            observation = self.sample_emission(state)
            predictions.append(observation)

            # Transition to the next state
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