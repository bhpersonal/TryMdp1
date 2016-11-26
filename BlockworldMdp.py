import MDP


def elem_add(t1, t2):
    return tuple([sum(x) for x in zip(t1, t2)])


class Blockworld1(MDP.MDP):

    def __init__(self):


        self.PENALTY = -10.0
        self.REWARD = 1.0

        self.actions = [0,1,2,3] # left up right down

        self.action_directions = [(-1,0), (0,1), (1, 0), (0, -1)]



        self.happy_states = [(4, 3)]
        self.sad_states = [(3, 2)]
        self.block_states = [(2,2)]

        self.states = [(x,y)
                  for x in range(1,5)
                  for y in range(1,4)
                  if (x,y) not in self.block_states]

        self.transition_probs = dict([((old_state, action, new_state), prob)
                                 for old_state in self.states
                                 for action in self.actions
                                 for (prob, new_state) in self.get_possible_new_states_and_probabilities(old_state, action)])

        self.transition_states = dict([((old_state, action), [new_state for (_, new_state) in
                                                         self.get_possible_new_states_and_probabilities(old_state, action)])
                                  for old_state in self.states
                                  for action in self.actions])

    def get_possible_new_states_and_probabilities(self, old_state, action):

        expected = (0.8, elem_add(self.action_directions[action], old_state))
        oops1 = (0.1, elem_add(self.action_directions[(action + 1) % len(self.actions)] , old_state))
        oops2 = (0.1, elem_add(self.action_directions[(action - 1 + len(self.actions)) % len(self.actions)], old_state))

        all = [expected, oops1, oops2]
        feasible = [(prob, state) for (prob, state) in all if state in self.states]
        prob_sum = sum([prob for (prob, state) in feasible])
        normalized = [(prob / prob_sum, state) for (prob,state) in feasible]

        return normalized


    def get_possible_new_states(self, old_state, action):
        return self.transition_states[(old_state, action)]

    def prob_func(self, old_state, action, new_state):

        return self.transition_probs[(old_state, action, new_state)]

    def reward_func(self, old_state, action, new_state):

        # Assume cartesian coordinate system
        if new_state in self.happy_states: return self.REWARD
        if new_state in self.sad_states: return self.PENALTY

        return 0

    def render(self, policy, values):

        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()

        plt.ylim([1,4])
        plt.xlim([1,5])

        for x in range(1, 6):
            plt.plot([1, 5], [x, x], color='k', linestyle='-', linewidth=2)

        for y in range(1, 5):
            plt.plot([y, y], [1, 4], color='k', linestyle='-', linewidth=2)

        for block in self.block_states:
            plt.plot([block[0], block[0] + 1], [block[1], block[1] + 1], color='k', linestyle='-', linewidth=2)
            plt.plot([block[0], block[0] + 1], [block[1] + 1, block[1]], color='k', linestyle='-', linewidth=2)

        for block in self.happy_states:
            plt.text(block[0] + 0.1, block[1] + 0.1, "+" + str(self.REWARD), fontsize=14, color = "red")

        for block in self.sad_states:
            plt.text(block[0] + 0.1, block[1] + 0.1, str(self.PENALTY), fontsize=14, color = "red")


        for state in self.states:
            action = policy[state]
            direction = self.action_directions[action]

            ax.arrow(state[0] + 0.5, state[1] + 0.5, direction[0] * 0.2, direction[1] * 0.2, head_width=0.05, head_length=0.1, fc='k', ec='k')

            plt.text(state[0] + 0.6, state[1] + 0.3, str(round(values[state], 2)), color="green", fontsize=14)

        plt.show()






