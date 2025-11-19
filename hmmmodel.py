import numpy as np
from math import log
class HiddenMarkovModel:
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
        state = self.__get_start_state()
        emission = self.__get_emission(state)
        self.emissions.append(emission)
        self.states.append(state)

        for _ in range(0, n_samples):
            state = self.__get_next_state(state)
            emission = self.__get_emission(state)

            self.states.append(state)
            self.emissions.append(emission)
        return np.array(self.emissions), np.array(self.states)


    def __get_start_state(self):
        return np.random.choice(self.__av_states, p = self.startprob_)

    def __get_emission(self, state):
        return np.random.choice(self.__emissions, p = self.emissionprob_[state])

    def __get_next_state(self, state):
        return np.random.choice(self.__av_states, p = self.transmat_[state])
