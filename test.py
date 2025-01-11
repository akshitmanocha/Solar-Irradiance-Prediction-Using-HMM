import numpy as np

class InfiniteHiddenMarkovModel():
    def __init__(self, alpha, beta, gamma, beta0, gamma0):
        if any(param <= 0 for param in [alpha, beta, gamma, beta0, gamma0]):
            raise ValueError("All hyperparameters must be positive")
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.beta0 = beta0
        self.gamma0 = gamma0
        
        # Initialize count structures
        self.transition_counts = {}
        self.oracle_counts = {}
        self.emission_counts = {}
        self.emission_oracle_counts = {}
        self.num_states = 0
        self.num_emissions = 0
        self.state_sequence = []
        
    def initialize_state(self):
        """Initialize a new state with proper count structures"""
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
        """Sample next state following the two-level HDP hierarchy"""
        try:
            # Handle initial or invalid state
            if current_state is None or current_state not in self.transition_counts:
                new_state = self.initialize_state()
                self.oracle_counts[new_state] = 1
                return new_state

            # Calculate total transitions from current state
            total_transitions = sum(self.transition_counts[current_state].values())
            
            # Calculate denominator including self-transition bias
            denominator = total_transitions + self.beta + self.alpha
            if denominator == 0:
                raise ValueError(f"Zero denominator in sample_next_state for state {current_state}")

            # Calculate probabilities for existing transitions
            probs = {}
            for state, count in self.transition_counts[current_state].items():
                p = (count + (self.alpha if state == current_state else 0)) / denominator
                probs[state] = p

            # Probability of consulting oracle
            prob_oracle = self.beta / denominator

            # Sample decision
            if np.random.rand() < prob_oracle:
                return self.consult_oracle()
            
            # Sample from existing transitions
            if not probs:  # If no existing transitions
                return self.consult_oracle()
                
            states = list(probs.keys())
            probabilities = list(probs.values())
            # Normalize probabilities
            probabilities = np.array(probabilities) / sum(probabilities)
            return np.random.choice(states, p=probabilities)
            
        except Exception as e:
            raise Exception(f"Error in sample_next_state: {str(e)}")

    def consult_oracle(self):
        """Consult the oracle DP"""
        try:
            total_oracle = sum(self.oracle_counts.values())
            if total_oracle + self.gamma == 0:
                raise ValueError("Zero denominator in consult_oracle")

            # Calculate probability of new state
            prob_new = self.gamma / (total_oracle + self.gamma)
            
            if np.random.rand() < prob_new:
                # Create new state
                new_state = self.initialize_state()
                self.oracle_counts[new_state] = 1
                return new_state
                
            # Sample from existing states with normalized probabilities
            states = list(self.oracle_counts.keys())
            probabilities = [count / (total_oracle + self.gamma) 
                           for count in self.oracle_counts.values()]
            # Ensure probabilities sum to 1
            probabilities = np.array(probabilities) / sum(probabilities)
            
            selected_state = np.random.choice(states, p=probabilities)
            self.oracle_counts[selected_state] += 1
            return selected_state
            
        except Exception as e:
            raise Exception(f"Error in consult_oracle: {str(e)}")

    def gibbs_sampling(self, observations, num_iter=1000):
        """Gibbs sampling with detailed error handling"""
        try:
            if not observations:
                raise ValueError("Observations sequence cannot be empty")
                
            T = len(observations)
            print(f"Initializing {T} states...")
            
            # Initialize state sequence
            self.state_sequence = []
            for _ in range(T):
                new_state = self.initialize_state()
                self.state_sequence.append(new_state)
                self.oracle_counts[new_state] = 1
            
            # Initialize emission counts
            print("Initializing emission counts...")
            for t in range(T):
                state = self.state_sequence[t]
                observation = observations[t]
                if observation not in self.emission_counts[state]:
                    self.emission_counts[state][observation] = 0
                self.emission_counts[state][observation] += 1
                
            # Initialize transition counts
            print("Initializing transition counts...")
            for t in range(T-1):
                curr_state = self.state_sequence[t]
                next_state = self.state_sequence[t+1]
                if next_state not in self.transition_counts[curr_state]:
                    self.transition_counts[curr_state][next_state] = 0
                self.transition_counts[curr_state][next_state] += 1

            print(f"Starting {num_iter} Gibbs iterations...")
            for iteration in range(num_iter):
                for t in range(T):
                    # Get current state and neighbors
                    current_state = self.state_sequence[t]
                    prev_state = self.state_sequence[t - 1] if t > 0 else None
                    
                    # Remove counts
                    if prev_state is not None:
                        if current_state in self.transition_counts[prev_state]:
                            self.transition_counts[prev_state][current_state] -= 1
                            if self.transition_counts[prev_state][current_state] == 0:
                                del self.transition_counts[prev_state][current_state]
                    
                    if observations[t] in self.emission_counts[current_state]:
                        self.emission_counts[current_state][observations[t]] -= 1
                        if self.emission_counts[current_state][observations[t]] == 0:
                            del self.emission_counts[current_state][observations[t]]
                    
                    # Sample new state
                    new_state = self.sample_next_state(prev_state)
                    self.state_sequence[t] = new_state
                    
                    # Update counts
                    if prev_state is not None:
                        if new_state not in self.transition_counts[prev_state]:
                            self.transition_counts[prev_state][new_state] = 0
                        self.transition_counts[prev_state][new_state] += 1
                    
                    if observations[t] not in self.emission_counts[new_state]:
                        self.emission_counts[new_state][observations[t]] = 0
                    self.emission_counts[new_state][observations[t]] += 1

                if iteration % 100 == 0:
                    print(f"Completed iteration {iteration}")
                    
        except Exception as e:
            raise Exception(f"Error in gibbs_sampling: {str(e)}")

    def particle_filtering(self, observations, num_particles=100):
        """
        Particle filtering for likelihood computation
        """
        if not observations:
            raise ValueError("No observations provided for particle filtering")
            
        T = len(observations)
        particles = [self.initialize_state() for _ in range(num_particles)]
        log_likelihood = 0
        
        for t in range(T):
            weights = np.zeros(num_particles)
            
            # Calculate particle weights
            for i, particle in enumerate(particles):
                emission_probs = self.emission_counts.get(particle, {})
                total_emissions = sum(emission_probs.values())
                
                if observations[t] in emission_probs:
                    prob = emission_probs[observations[t]] / (total_emissions + self.beta0)
                else:
                    prob = self.beta0 / (total_emissions + self.beta0)
                weights[i] = prob
                
            # Normalize weights
            weights = weights / np.sum(weights)
            log_likelihood += np.log(np.mean(weights))
            
            # Resample particles
            indices = np.random.choice(num_particles, size=num_particles, p=weights)
            particles = [particles[i] for i in indices]
            
            # Propagate particles
            particles = [self.sample_next_state(p) for p in particles]
            
        return log_likelihood

    def update_counts(self, current_state, next_state):
        """Update transition counts"""
        if next_state not in self.transition_counts[current_state]:
            self.transition_counts[current_state][next_state] = 0
        self.transition_counts[current_state][next_state] += 1

    def update_emission_counts(self, state, emission):
        """Update emission counts"""
        if emission not in self.emission_counts[state]:
            self.emission_counts[state][emission] = 0
        self.emission_counts[state][emission] += 1
    
    def sample_emission(self, state):
        """
        Sample an emission from a given state using the hierarchical emission distribution
        """
        try:
            if state not in self.emission_counts:
                raise ValueError(f"Invalid state {state}")
                
            # Get emission counts for this state
            state_emissions = self.emission_counts[state]
            total_emissions = sum(state_emissions.values())
            
            # Calculate probability of new emission vs existing emissions
            denominator = total_emissions + self.beta0
            prob_new = self.beta0 / denominator
            
            # Decide whether to generate new emission or sample existing
            if np.random.rand() < prob_new:
                # Sample from emission oracle (base distribution)
                # For discrete observations, we'll sample from existing unique emissions
                all_emissions = set()
                for state_emissions in self.emission_counts.values():
                    all_emissions.update(state_emissions.keys())
                if not all_emissions:
                    return "obs_1"  # Default if no emissions exist
                return np.random.choice(list(all_emissions))
            
            # Sample from existing emissions
            emissions = list(state_emissions.keys())
            probs = [count / denominator for count in state_emissions.values()]
            # Normalize probabilities
            probs = np.array(probs) / sum(probs)
            
            return np.random.choice(emissions, p=probs)
            
        except Exception as e:
            raise Exception(f"Error in sample_emission: {str(e)}")

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
        # Initialize the hyperparameters
        alpha = 1.0    # Moderate self-transitions
        beta = 3.0     # Some transition exploration
        gamma = 10.0   # Few new states
        beta0 = 1.0    # Standard emission variety
        gamma0 = 1.0   # Limited emission vocabulary
        
        # Print hyperparameters before optimization
        print("Hyperparameters before optimization:")
        print(f"alpha = {alpha}, beta = {beta}, gamma = {gamma}, beta0 = {beta0}, gamma0 = {gamma0}")

        print("\nInitializing IHMM...")
        ihmm = InfiniteHiddenMarkovModel(
            alpha=alpha,    # Moderate self-transitions
            beta=beta,      # Some transition exploration
            gamma=gamma,    # Few new states
            beta0=beta0,    # Standard emission variety
            gamma0=gamma0   # Limited emission vocabulary
        )

        observations = ["obs_1", "obs_2", "obs_1", "obs_2", "obs_1","obs_2","obs_1","obs_2", "obs_1","obs_2","obs_1"]
        print(f"\nObservations: {observations}")

        print("\nPerforming Gibbs sampling...")
        ihmm.gibbs_sampling(observations, num_iter=5000)
        print("Hidden state sequence:", ihmm.state_sequence)

        # Here, you can modify or optimize hyperparameters if needed. 
        # For example, let's say we change them after optimization:
        # (This is just a placeholder; implement your actual optimization logic here)
        alpha = 0.8  # Example of optimized hyperparameter
        beta = 2.5
        gamma = 8.0
        beta0 = 0.9
        gamma0 = 0.8

        # Print hyperparameters after optimization
        print("\nHyperparameters after optimization:")
        print(f"alpha = {alpha}, beta = {beta}, gamma = {gamma}, beta0 = {beta0}, gamma0 = {gamma0}")

        print("\nPredicting future observations...")
        initial_state = ihmm.state_sequence[0]
        predictions = ihmm.predict(initial_state, steps=10)
        print("Predicted observations:", predictions)

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()