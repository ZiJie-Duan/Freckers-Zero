from freckers_gym import MctsAcc
from freckers_gym import Player


mcts = MctsAcc()

print(mcts.get_action_space(Player.Red))
s,sn,r,end,valid = mcts.step(Player.Red, 0,1,1,1,False)
from pprint import pprint

pprint(s)
pprint(sn)
print(r)
print(end)
print(valid)

