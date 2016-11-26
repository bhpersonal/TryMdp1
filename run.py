import MDP
import BlockworldMdp
import ValueIteration

mdp = BlockworldMdp.Blockworld1()

policy, values = ValueIteration.generate_policy(mdp, 0.7, 0.0001, 1000000)

mdp.render(policy, values)