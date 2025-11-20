import numpy as np
from math import log
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

    def score(self, X, state_sequence):
        self.__validate_matrices()
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

    def predict(self, X):
        self.__validate_matrices()
        previous_chances = []
        states = []
        self.__predict_starting_state(X, previous_chances, states)

        for emission in X[1:]:
            temp_previous_chances = []

            for state in self.__av_states:
                temp_chances = []

                for state_2 in self.__av_states:
                    sum_probs = previous_chances[state_2] + self.__safe_log(self.transmat_[state_2][state]) + self.__safe_log(self.emissionprob_[state][emission])
                    temp_chances.append(sum_probs)

                temp_previous_chances.append((max(temp_chances)))

            previous_chances = temp_previous_chances
            states.append(previous_chances.index(max(previous_chances)))

        return states

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

    def newPredict(self, X):
        n_states = len(self.__av_states)
        n_observations = len(X)
        prob_values = np.full((n_states, n_observations), float("-inf"))
        states_matrix = np.zeros((n_states, n_observations))


        for state in range(0, n_states):
            prob_values[state, 0] = self.__safe_log(self.startprob_[state]) + self.__safe_log(self.emissionprob_[state][X[0]])

        for t in range(1, n_observations):
            for state in self.__av_states:
                max_prob = float("-inf")
                best_state = 0
                for prev_state in self.__av_states:
                    prob = prob_values[prev_state, t-1] + self.__safe_log(self.transmat_[prev_state][state]) + self.__safe_log(self.emissionprob_[state][X[t]])

                    if prob > max_prob:
                        max_prob = prob
                        best_state = prev_state
                prob_values[state, t] = max_prob
                states_matrix[state, t] = best_state
        return states_matrix

    def __get_random_start_state(self):
        return np.random.choice(self.__av_states, p = self.startprob_)

    def __get_random_emission(self, state):
        return np.random.choice(self.__emissions, p = self.emissionprob_[state])

    def __get_random_next_state(self, state):
        return np.random.choice(self.__av_states, p = self.transmat_[state])
