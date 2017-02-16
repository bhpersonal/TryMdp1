

class MDP (object):

    def __init__(self, states, actions, get_possible_new_states, prob_func, reward_func):

        self.states = states
        self.actions = actions
        self.get_possible_new_states = get_possible_new_states
        self.prob_func = prob_func
        self.reward_func = reward_func


    def verify_probabilities(self):

        for old_state in self.states:
            for action in self.actions:

                new_states = self.get_possible_new_states(old_state, action)
                new_state_probs = [self.prob_func(old_state, action, new_state) for new_state in new_states]
                total_prob = sum(new_state_probs)

                print (str(old_state) + " do " + str(action) + ": " + str(new_states) + ", " + str(new_state_probs))

                assert abs(1.0 - total_prob) < 0.0001, "sum of probabilities for P(s' | s, a) != 1.0, for state " + str(old_state) + ", action " + str(action)


    def print_transitions(self):
        for old_state in self.states:
            for action in self.actions:
                new_states = self.get_possible_new_states(old_state, action)

                print (str(old_state) + " do " + str(action) + ": " + str(new_states))



