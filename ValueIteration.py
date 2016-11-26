def argmax(items, func):
    if items == []: return None

    max_item = items[0]
    max_value = func(items[0])

    for item in items:

        value = func(item)
        if value > max_value:
            max_value = value
            max_item = item

    return max_item

def generate_policy(mdp, decay, max_error, max_iterations):

    # naive initialization of V
    vs = []

    vs.append(dict([(state, 0) for state in mdp.states]))
    k = 0


    def v_func (old_state, action):
        ## Bellman equation (sort of)
        return sum([ (mdp.prob_func(old_state, action, new_state) * (mdp.reward_func(old_state, action, new_state)  + decay * vs[k][new_state] ) )
                     for new_state in mdp.get_possible_new_states(old_state, action)])


    while k < max_iterations:

        vs.append(dict([(state, 0) for state in mdp.states]))

        for (si, old_state) in enumerate(mdp.states):

            vs[k+1][old_state] = max([ v_func(old_state, action) for action in mdp.actions])

        k += 1

        if all(abs(vs[k][state] - vs[k-1][state]) <= max_error for state in mdp.states ):
            break

    # generate policy
    policy = dict([ (state, (argmax(mdp.actions, lambda action: v_func(state, action)))) for state in mdp.states])

    return policy, vs[k]