import MDP
import BlockworldMdp
import ValueIteration

russel_norvig_example = BlockworldMdp.Blockworld1(
    width=4,height=3,
    goal_positions=[(4, 3)],
    penalty_positions=[(4,2)],
    obstacle_positions=[(2,2)],
    time_cost=0.1)


big_example = BlockworldMdp.Blockworld1(
    width=10,height=9,
    goal_positions=[(5, 6)],
    penalty_positions=[(7,3), (2, 5), (3,6), (8,4), (1,2), (9,5)],
    obstacle_positions=[(5, 4),(2, 8),(3, 8),(4, 8), (3, 3), (7, 1), (7,2)],
    time_cost=0.5,
    penalty_value=-10.0)


mdp =russel_norvig_example

policy, values = ValueIteration.generate_policy(mdp, 0.9, 0.0001)

mdp.render(policy, values)