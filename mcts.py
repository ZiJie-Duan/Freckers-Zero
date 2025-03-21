from freckers_gym import MctsAcc
from freckers_gym import Player
import math
from fnet import Conv3DStack, inference
import torch

model = Conv3DStack()

class MCTSFreckerZero:
    def __init__(self, p, game=None) -> None:
        self.game = game
        self.n = 0 # add 1 when backp
        self.w = 0 # add v when backp
        self.q = 0 # 1 / self.n * backp v_acc 
        self.p = p # from parent nn

        self.v = 0 # from nn, probability for win
        self.u = 0 # c * p * sqrt(all_n) / (1+n)
        self.children = [] #(child, (r,c,nr,nc)/(true,0,0,0)) 
        self.c = 2

    def run(self, step = 100):
        for i in range(step):
            self.simu(self.game.dclone(), Player.Red)

    def go(self):
        pass

    def simu(self, game, player):
        actions = {}
        tensor_surface_order = {}

        if self.n == 0:
            for space in game.get_action_space(player):
                actions[space[0]] = space[1:]
            
            game_py_tensor, fogs = game.get_game_tensor(player)
            for i in range(2,len(game_py_tensor)):
                for r,c in fogs:
                    if game_py_tensor[i][r][c] == 1:
                        tensor_surface_order[(r,c)] = i - 2 # shift to the 0 surface

            game_tensor = torch.tensor(game_py_tensor)
            result, v = inference(model, game_tensor)
            result = result[0]
            v = v[0]

            for loc, action_group in actions.items():
                for a in action_group:
                    self.children.append(
                        (MCTSFreckerZero(result[tensor_surface_order[loc]][loc[0]][loc[1]]),
                        (loc[0], loc[1], a[0], a[1]))
                    )

            self.children.append(
                (MCTSFreckerZero(v[1]),(True,0,0,0))
            )

            max = -999
            min = 999
            for c, a in self.children:
                if c.p > max:
                    max = c.p
                if c.p < min:
                    min = c.p

            for c, a in self.children:
                c.p = (c.p - min)/(max - min)

            game.pprint()
            self.n += 1
            return v[0]

        else:
            u_total = 0
            for c in self.children:
                u_total += c[0].n

            max = -999
            max_i = 0
            for i, child_With_action in enumerate(self.children):
                c = child_With_action[0]
                v = (c.w / (c.n + 0.000001)) + (c.c * c.p * (math.sqrt(u_total)) / (1+c.n))
                if max < v:
                    max = v
                    max_i = i
                print(f"CAL PUBT: c.w:{c.w/(c.n+0.01)}, c.c:{c.c}, c.p:{c.p}")
                print(f"PUBT: {v} , action: {child_With_action[1]}")
            
            print(f"Choose: {self.children[max_i]}")
            # backp
            r,c,rn,cn = self.children[max_i][1]
            if r == True:
                s,sn,r,end,valid = game.step(player,0,0,0,0,True)
            else: 
                s,sn,r,end,valid = game.step(player,r,c,rn,cn,False)
            v = self.children[max_i][0].simu(game, Player.Red if player == Player.Blue else Player.Blue)
            self.w += v
            self.n += 1
            return v


game = MctsAcc()
mcts = MCTSFreckerZero(1,game)
mcts.run(100)



