from freckers_gym import Game, Player, Action
import time

g = Game()

g = Game()
start_time = time.time()
pr = Player.Red
pb = Player.Blue
g.pprint()

_,_,_,_,_,v = g.step(pr, Action(0,2,4))
g.pprint()
_,_,_,_,_,v = g.step(pr, Action(True))
g.pprint()

_,_,_,_,_,v = g.step(pb, Action(7,6,0))
_,_,_,_,_,v = g.step(pb, Action(7,3,0))
_,_,_,_,_,v = g.step(pb, Action(grow=True))
g.pprint()

print(f"执行时间: {time.time() - start_time} 秒")