
import random
import BlockworldMdp

world = BlockworldMdp.Blockworld1(
    width=4,height=3,
    goal_positions=[(4, 3)],
    penalty_positions=[(4,2)],
    obstacle_positions=[(2,2)],
    time_cost=-0.1)




qqq = None


def choose_action(state, qtable):

    qs = qtable[state]

    max_q = max(qs)

    max_actions = [ i for (i, q) in enumerate(qs) if q == max_q]

    action = random.choice(max_actions)
    return action


#https://studywolf.wordpress.com/2012/11/25/reinforcement-learning-q-learning-and-exploration/
#http://teaching.csse.uwa.edu.au/units/CITS4211/Lectures/wk7.pdf
#http://mnemstudio.org/path-finding-q-learning-tutorial.htm
qtable = dict( ( (s, [0.0 for a in world.actions]) for s in world.states))
random_epsilon = 0.1
learning_rate = 0.1
discount_factor = 0.9

for episode in range(0, 100):

    current_state = random.choice(world.states)


    while not current_state == world.terminal_state:


        old_state = current_state
        action = choose_action(current_state,qtable)
        new_state = world.perform_action(old_state, action)

        new_reward = world.reward_func(old_state,action,new_state)


        max_q = max(qtable[new_state])

        #qtable[old_state][action] = reward + learning_rate * max_q
        #qtable[old_state][action] += learning_rate * ( old_reward + max_q - qtable[old_state][action])
        qtable[old_state][action] = qtable[old_state][action] + learning_rate * ( new_reward + discount_factor * max_q -  qtable[old_state][action] )


        current_state = new_state
        print (str(old_state) + ", " + str(action) + ", " + str(current_state) + ": " + str(new_reward) + "   ==>  " + str(qtable[old_state][action]))


    policy = dict((s,qtable[s].index(max(qtable[s])))     for s in world.states)
    values = dict((s, max(qtable[s]))     for s in world.states)

    print (policy)


    world.render(policy, values)






