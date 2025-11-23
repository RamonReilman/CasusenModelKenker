import numpy as np
from math import log
from scipy.special import logsumexp
class HiddenMarkovModel:
    __SCALE = 10
    def __init__(self, n_components, n_features):
        self.startprob_ = None
        self.transmat_ = None
        self.emissionprob_ = None
        self.states = []
        self.emissions = []

        self.n_components = n_components
        self.__av_states = np.arange(0, n_components, 1)
        self.__emissions = np.arange(0, n_features, 1)
        self.n_features = n_features

    def __str__(self):
        return f"""
                {self.__repr__()}\n
                States: {np.array(self.states)}\n
                Emissions: {np.array(self.emissions)}\n
                """

    def __repr__(self):
        return f"HiddenMarkovModel(n_components:{self.n_components},n_features:{self.n_features})"

    def score(self, X, state_sequence = None):
        if state_sequence is not None:
            return self.__log_prob_with_states(X, state_sequence)

        return self.__log_prob_without_states(X)

    def __log_prob_without_states(self, X):
        n_obs = len(X)
        n_states = len(self.__av_states)

        prob_matrix = np.zeros((n_states, n_obs), dtype=float)
        prob_matrix_log = np.zeros((n_states, n_obs), dtype=float)

        for state in range(n_states):
            prob_matrix[state, 0] = (self.startprob_[state]  * self.emissionprob_[state][X[0]])
            prob_matrix_log[state, 0] = (self.__safe_log(self.startprob_[state]) + self.__safe_log(self.emissionprob_[state][X[0]]))
        for t in range(1, n_obs):
            for state_t in range(0, n_states):
                temp_probability = []
                temp_probability_log = []
                for state_t_minus in range(0, n_states):
                    prev_probability = prob_matrix[state_t_minus, t-1]
                    prev_prob_log = prob_matrix_log[state_t_minus, t-1]
                    state_t_minus_X_emission = self.emissionprob_[state_t][X[t]]
                    state_transition_prob = self.transmat_[state_t_minus][state_t]

                    temp_probability.append(prev_probability * state_t_minus_X_emission * state_transition_prob)
                    temp_probability_log.append(prev_prob_log + self.__safe_log(state_t_minus_X_emission) + self.__safe_log(state_transition_prob))

                prob_matrix[state_t, t] = sum(temp_probability)
                prob_matrix_log[state_t, t] = logsumexp(temp_probability_log)

        return np.sum(prob_matrix[:, -1]), logsumexp(prob_matrix_log[:, -1])


    def __log_prob_with_states(self, X, state_sequence):
        log_score = 0.0
        state_prob = self.startprob_[state_sequence[0]]
        emission_prob = self.emissionprob_[state_sequence[0]][X[0]]
        log_score += log(state_prob)
        log_score += log(emission_prob)
        for (emission, prev_state, current_state) in zip(X[1:], state_sequence[:-1], state_sequence[1:]):
            state_prob = self.transmat_[prev_state][current_state]
            emission_prob = self.emissionprob_[current_state][emission]
            if state_prob == 0 or emission_prob == 0:
                return float("-inf")
            log_score += log(state_prob)
            log_score += log(emission_prob)

        return log_score

    def sample(self, n_samples):
        self.__validate_matrices()
        state = self.__get_random_start_state()
        emission = self.__get_random_emission(state)
        self.emissions.append(emission)
        self.states.append(state)

        for _ in range(0, n_samples):
            state = self.__get_random_next_state(state)
            emission = self.__get_random_emission(state)

            self.states.append(state)
            self.emissions.append(emission)
        return np.array(self.emissions), np.array(self.states)


    def __predict_starting_state(self, X, previous_chances, states):
        for state in self.__av_states:
            sum_probs = self.__safe_log(self.startprob_[state]) + self.__safe_log(self.emissionprob_[state][X[0]])
            previous_chances.append(sum_probs)
        states.append(previous_chances.index(max(previous_chances)))

    def __validate_matrices(self):
        if self.emissionprob_ is None or self.transmat_ is None or self.startprob_ is None:
            raise ValueError("One or more of the probability matrices (startprob_, emissionprob_, transmat_) are None")

    def __safe_log(self, p):
        return float("-inf") if p == 0 else log(p)

    def predict(self, X):

        # Setup matrices
        n_states = len(self.__av_states)
        n_observations = len(X)
        prob_values = np.full((n_states, n_observations), float("-inf"))
        states_matrix = np.zeros((n_states, n_observations), dtype=int)

        # Fill in first probability based on starting values
        for state in range(0, n_states):
            prob_values[state, 0] = self.__safe_log(self.startprob_[state]) + self.__safe_log(self.emissionprob_[state][X[0]])

        # Calculate probability values
        for t in range(1, n_observations):
            for state in self.__av_states:
                max_prob = float("-inf")
                best_state = 0
                for prev_state in self.__av_states:
                    prob = prob_values[prev_state, t-1] + self.__safe_log(self.transmat_[prev_state][state]) + self.__safe_log(self.emissionprob_[state][X[t]])

                    if prob > max_prob:
                        max_prob = prob
                        best_state = prev_state
                # Adds the highest probability of a point in time: t for a state
                prob_values[state, t] = max_prob
                # Adds the state at t-1 that gave emission at t the highest probability
                states_matrix[state, t] = best_state
        return np.array(self.__build_sequence_of_states(prob_values, states_matrix))

    def __build_sequence_of_states(self, probabilities, states):
        sequence = []
        final_best_state = self.__get_final_state(probabilities)
        sequence.append(final_best_state)
        for i in range(probabilities.shape[1]-1, 0 ,-1):
            final_best_state = states[final_best_state, i]
            sequence.append(final_best_state)

        return sequence[::-1]


    def __get_final_state(self, probabilities):
        return np.argmax(probabilities[:, -1])



    def __get_random_start_state(self):
        return np.random.choice(self.__av_states, p = self.startprob_)

    def __get_random_emission(self, state):
        return np.random.choice(self.__emissions, p = self.emissionprob_[state])

    def __get_random_next_state(self, state):
        return np.random.choice(self.__av_states, p = self.transmat_[state])
