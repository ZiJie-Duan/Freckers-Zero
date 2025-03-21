from freckers_gym import MctsAcc
from freckers_gym import Player
import math
from fnet import Conv3DStack, inference
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

model = Conv3DStack()

memory = [] # (game_tensor, )


class PointLoss(nn.Module):
    def __init__(self):
        super(PointLoss, self).__init__()
    def forward(self, img_p, prob_p, pi, value):
        loss = torch.tensor(0.0, device=img_p.device)
        for p in pi:
            if p[0]:
                term = (img_p[6][5][5] - p[1]) ** 2
            else:
                term = (img_p[p[0]][p[1]][p[2]] - p[3]) ** 2
            loss = loss + term
        loss = loss + prob_p
        return loss

def train(model, memory, num_epochs=30):
    
    # 双损失函数：图像用MSE，概率用交叉熵
    criterion = PointLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for st, pi, value in memory:
            st = torch.tensor(st).float().requires_grad_(True).unsqueeze(0)
            optimizer.zero_grad()
            result, v = model(st)
            result = result[0]
            v = v[0]
            total_loss = criterion(result, v, pi, value)

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {running_loss/len(memory):.4f}")

class MCTSFreckerZero:
    def __init__(self, p, game=None, player=Player.Red) -> None:
        self.game = game
        self.player = player
        self.n = 0 # add 1 when backp
        self.w = 0 # add v when backp
        self.q = 0 # 1 / self.n * backp v_acc 
        self.p = p # from parent nn

        self.children = [] #(child, (r,c,nr,nc)/(true,0,0,0)) 
        self.c = 2
        self.t = 1

        self.finished = False

    def run(self, step = 100):
        for i in range(step):
            if i > (step - 0):
                self.simu(self.game.dclone(), self.player, visual = True)
            else:
                self.simu(self.game.dclone(), self.player, visual = False)

    def go(self):
        children_pubt = []
        max = -999
        max_i = 0
        tt = sum([c[0].n**(1/self.t) for c in self.children])
        for i, child_With_action in enumerate(self.children):
            c = child_With_action[0]
            v = (c.n)**(1/self.t) / tt
            children_pubt.append(v)
            if max < v:
                max = v
                max_i = i

        st, fogs_info = self.game.get_game_tensor(self.player)
        surface_map = {}
        for loc, surface in fogs_info:
            surface_map[loc] = surface
        pi = [] # (surface, row, col, v) / (True, v)
        for i, data in enumerate(self.children):
            c, a = data
            if a[0] == True:
                pi.append((True, children_pubt[i]))
            else:
                pi.append((surface_map[(a[0], a[1])], a[2], a[3], children_pubt[i]))
        
        memory.append((st, pi, self.children[max_i][0].w / self.children[max_i][0].n))

        r,c,rn,cn = self.children[max_i][1]
        if r == True:
            s,sn,r,end,valid = self.game.step(self.player,0,0,0,0,True)
        else: 
            s,sn,r,end,valid = self.game.step(self.player,r,c,rn,cn,False)
        
        self.game.pprint()
        child = self.children[max_i][0]
        self.player = Player.Red if self.player == Player.Blue else Player.Blue
        self.n = child.n # add 1 when backp
        self.w = child.w # add v when backp
        self.q = child.q # 1 / self.n * backp v_acc 
        self.p = child.p # from parent nn

        self.children = child.children #(child, (r,c,nr,nc)/(true,0,0,0)) 
        self.c = child.c
        self.t = child.t



    def simu(self, game, player, visual = False):
        actions = {}
        surface_map = {}

        if self.n == 0:
            for space in game.get_action_space(player):
                actions[space[0]] = space[1:]
            
            game_py_tensor, fogs_info = game.get_game_tensor(player)
            for loc, surface in fogs_info:
                surface_map[loc] = surface
            game_tensor = torch.tensor(game_py_tensor).flip(dims=[1, 2])
            result, v = inference(model, game_tensor)
            print(result.shape)
            result = np.rot90(result, 2)[0]
            v = v[0]

            for loc, action_group in actions.items():
                for a in action_group:
                    self.children.append(
                        (MCTSFreckerZero(result[surface_map[loc]][loc[0]][loc[1]]),
                        (loc[0], loc[1], a[0], a[1]))
                    )

            self.children.append(
                (MCTSFreckerZero(result[6][5][5]),(True,0,0,0))
            )

            max = -999
            min = 999
            for c, a in self.children:
                if c.p > max:
                    max = c.p
                if c.p < min:
                    min = c.p

            for c, a in self.children:
                c.p = (c.p - min)/(max - min + 0.000001)

            if visual == True:
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
                if False:
                    print(f"CAL PUBT: c.w:{c.w/(c.n+0.01)}, c.c:{c.c}, c.p:{c.p}")
                    print(f"PUBT: {v} , action: {child_With_action[1]}")
            
            #print(f"Choose: {self.children[max_i]}")
            # backp
            r,c,rn,cn = self.children[max_i][1]
            if r == True:
                s,sn,r,end,valid = game.step(player,0,0,0,0,True)
            else: 
                s,sn,r,end,valid = game.step(player,r,c,rn,cn,False)

            if end:
                self.finished = True
                v = r
            else:
                v = self.children[max_i][0].simu(game, Player.Red if player == Player.Blue else Player.Blue, visual)
            self.w += v
            self.n += 1
            return v


for _ in range(10):
    memory = []

    for _ in range(3):
        game = MctsAcc()
        mcts = MCTSFreckerZero(1,game)

        for i in range(60):
            print(f"Step {i+1}")
            mcts.run(250)
            if mcts.finished:
                break
            mcts.go()

    train(model, memory, 20)
    model.eval()

    