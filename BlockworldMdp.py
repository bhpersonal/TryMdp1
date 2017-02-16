import MDP
from itertools import groupby
from operator import itemgetter
import random
import numpy.random as rnd


def elem_add(t1, t2):
    return tuple([sum(x) for x in zip(t1, t2)])


class Blockworld1(MDP.MDP):

    def __init__(self, width = 4, height = 3, time_cost = -0.1, penalty_value = -1.0, goal_value = 1.0, goal_positions=[(4, 3)], penalty_positions=[(4, 2)], obstacle_positions=[(2,2)]):

        self.width = width
        self.height = height


        self.penalty_value = penalty_value
        self.goal_value = goal_value
        self.time_cost = time_cost

        self.actions = [0,1,2,3] # left up right down

        self.action_directions = [(-1,0), (0,1), (1, 0), (0, -1)]

        self.goal_states = goal_positions
        self.fail_states = penalty_positions
        self.obstacle_states = obstacle_positions

        self.terminal_state = (-1,-1)

        self.states = [(x,y)
                       for x in range(1,width+1)
                       for y in range(1,height+1)
                       if (x,y) not in self.obstacle_states
                       ]
        self.states.append(self.terminal_state)

        self.transition_probs = dict([((old_state, action, new_state), prob)
                                 for old_state in self.states
                                 for action in self.actions
                                 for (prob, new_state) in self.get_possible_new_states_and_probabilities(old_state, action)])

        self.transition_states = dict([((old_state, action), [new_state for (_, new_state) in
                                                         self.get_possible_new_states_and_probabilities(old_state, action)])
                                  for old_state in self.states
                                  for action in self.actions])



    def get_possible_new_states_and_probabilities(self, old_state, action):

        if old_state in self.goal_states:
            return [(1.0, self.terminal_state)]
        if old_state in self.fail_states:
            return [(1.0, self.terminal_state)]

        expected = (0.8, elem_add(self.action_directions[action], old_state))
        oops1 = (0.1, elem_add(self.action_directions[(action + 1) % len(self.actions)] , old_state))
        oops2 = (0.1, elem_add(self.action_directions[(action - 1 + len(self.actions)) % len(self.actions)], old_state))

        all = [expected, oops1, oops2]
        prob_stay = 0.0
        if expected[1] not in self.states:
            prob_stay += expected[0]
            all.remove(expected)
        if oops1[1] not in self.states:
            prob_stay += oops1[0]
            all.remove(oops1)
        if oops2[1] not in self.states:
            prob_stay += oops2[0]
            all.remove(oops2)

        if prob_stay > 0.0:
            all.append((prob_stay, old_state ))

        return all

        #feasible = [(prob, state) for (prob, state) in all if state in self.states]
        #prob_sum = sum([prob for (prob, state) in feasible])
        #normalized = [(prob / prob_sum, state) for (prob,state) in feasible]

        # # invalid actions, stay in same state
        # all = [(prob, (cell if cell in self.states else old_state)) for (prob, cell) in all]
        #
        # # merge probabilities for identical outcomes
        # all = sorted(all, key=itemgetter(1))
        # grps = list(groupby(all, key=itemgetter(1))) #merge by cell
        # grps = [(key, list(iter(items))) for (key, items) in grps]
        # all = [(sum([p for p, _ in items]), cell) for (cell, items) in grps] # recreate list summing prob foreach cell
        #
        # return all


    def get_possible_new_states(self, old_state, action):
        return self.transition_states[(old_state, action)]

    def prob_func(self, old_state, action, new_state):

        return self.transition_probs[(old_state, action, new_state)]

    def reward_func(self, old_state, action, new_state):

        # Assume cartesian coordinate system
        if new_state in self.goal_states: return self.goal_value
        if new_state in self.fail_states: return self.penalty_value

        return self.time_cost

    def perform_action(self, old_state, action):

        tx = [(s2, self.transition_probs[(s1, a, s2)]) for (s1, a, s2) in self.transition_probs.keys() if s1 == old_state and a == action]

        s2 = [s2 for (s2,p) in tx]
        p = [p for (s2, p) in tx]

        new_state = s2[rnd.choice(range(0, len(s2)), p=p)]
        return new_state


    def render(self, policy, values):

        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        import numpy as np

        fig, ax = plt.subplots()


        plt.xlim([1, self.width+1])
        plt.ylim([1, self.height + 1])

        for x in range(1, self.height+1):
            plt.plot([1, self.width+1], [x, x], color='k', linestyle='-', linewidth=2)

        for y in range(1, self.width+1):
            plt.plot([y, y], [1, self.height+1], color='k', linestyle='-', linewidth=2)

        for block in self.obstacle_states:
            #plt.plot([block[0], block[0] + 1], [block[1], block[1] + 1], color='k', linestyle='-', linewidth=2)
            #plt.plot([block[0], block[0] + 1], [block[1] + 1, block[1]], color='k', linestyle='-', linewidth=2)
            ax.add_patch(patches.Rectangle((block[0], block[1],), 1, 1, color="darkgrey"))

        for block in self.goal_states:
            plt.text(block[0] + 0.1, block[1] + 0.1, "+" + str(self.goal_value), fontsize=14, color ="red")
            ax.add_patch(patches.Rectangle((block[0], block[1],),1,1, color="green", alpha=0.5))

        for block in self.fail_states:
            plt.text(block[0] + 0.1, block[1] + 0.1, str(self.penalty_value), fontsize=14, color ="red")
            ax.add_patch(patches.Rectangle((block[0], block[1],), 1, 1, color="orange", alpha=0.5))


        for state in self.states:
            action = policy[state]
            direction = self.action_directions[action]

            ax.arrow(state[0] + 0.5, state[1] + 0.5, direction[0] * 0.2, direction[1] * 0.2, head_width=0.05, head_length=0.1, fc='k', ec='k')

            plt.text(state[0] + 0.6, state[1] + 0.3, str(round(values[state], 2)), color="green", fontsize=14)

        plt.show()






