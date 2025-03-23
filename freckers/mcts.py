from math import sqrt
from freckers_gym import RSTK
from deep_frecker import DeepFrecker
from deep_frecker import DataRecord
import numpy as np
import copy
from game import Game

print("start")
deep_frecker = DeepFrecker()
data_record = DataRecord(file=r"C:\Users\lucyc\Desktop\freckers_zero\data.h5", save_interval=300)

class MctsConfig:
    def __init__(self) -> None:
        self.c = 2
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

class MCTS:
    def __init__(self, prob, action, config, game=None, player=0) -> None:
        self.game = game
        self.player = player # 0/1
        self.config = config
        self.action = action # (r,c,nr,nc,grow)
        self.n = 0 # add 1 when backp
        self.w = 0 # add v when backp
        self.q = 0 # 1 / self.n * backp v_acc 
        self.p = prob # from parent nn

        self.children = [] 
        self.meta_value = 0

    def select(self):
        max = -999
        max_child = None

        # U(s,a) = C_puct * P(s,a) * Sqrt(Sum(N(s,b)_b_for_all_children)) / (1 + N(s,a)) 
        # all_num = Sqrt(Sum(N(s,b)_b_for_all_children))
        all_num = sqrt(sum([c.n for c in self.children]))

        debug_rec = []

        for child in self.children:

            # puct = Q(s,a) + U(s,a)
            u = self.config.c * child.p * (all_num) / (1 + child.n)
            puct = child.q + u
            if max < puct:
                max = puct
                max_child = child
            
            debug_rec.append((puct, child.action))
        
        if self.config.visulze:
            print("\n\ndebug_rec Player:", self.player)
            for i in range(len(debug_rec)):
                print(debug_rec[i])
        
        return max_child


    def expand(self, action_prob, rstk):

        for actions in rstk.get_action_space(self.player):
            base_loc = actions[0] # the location of the chess
            for action in actions[1:]: # loop to get the location where the chess will be placed
                
                self.children.append(
                    MCTS(
                        prob = action_prob[DeepFrecker.loc_trans(action[0], action[1])][base_loc[0]][base_loc[1]],
                        action = (base_loc[0], base_loc[1], action[0], action[1], False),
                        config = self.config,
                        player = 0 if self.player == 1 else 1
                        )
                )

        self.children.append(
            MCTS(
                prob = action_prob[5][5][5], # fix the location of grow probability
                action = (0, 0, 0, 0, True), # grow
                config = self.config,
                player = 0 if self.player == 1 else 1
                )
        )
            

    def simu(self, game):

        if self.n == 0:
            # eval
            gameboard = game.get_gameboard()
            action_prob, value = deep_frecker.run(gameboard.copy(), self.player)
            rstk = RSTK(gameboard.copy())
            # expand
            self.expand(action_prob, rstk)
            # backp
            self.meta_value = value[0] # value has two value, first is win, second is lose
            self.n += 1
            self.w += value[0]
            self.q = value[0] / self.n
            return value[0],0 # return estimated value and, but no reward

        else:
            # selection
            child = self.select()
            # move
            s,r,sn,end = game.step(self.player, *child.action)

            if end:
                self.n += 1
                self.w += r
                self.q = (r + self.meta_value) / self.n
                return r,r # if end, return the reward instead of the estimated value
            else:
                # go deeper
                value,r = child.simu(game)
                # backp
                self.n += 1
                self.w += value
                self.q = (value + self.meta_value) / self.n

            return value,r # if not end, return the estimated value and the child's reward
    

    def run_simu(self, rounds):
        for _ in range(rounds):
            game = copy.deepcopy(self.game)
            self.simu(game)


    def move(self):

        # Pi(a,s) = N(s,a)**(1/t) / Sum(N(s,b)**(1/t))
        # t_v = Sum(N(s,b)**(1/t))
        t_v = sum([c.n**(1/self.config.t) for c in self.children])
        pi = []

        max = -999
        max_child = 0
        v_order_rec = [] # get the probability of each action

        for i, child in enumerate(self.children):
            v = (child.n)**(1/self.config.t) / (t_v + self.config.small)
            v_order_rec.append(v)

            # build up the strategy Pi(a,s)
            pi.append(list(child.action))
            if max < v:
                max = v
                max_child = child

        # normalize the probability
        v_order_rec = np.array(v_order_rec)
        v_order_rec = v_order_rec / np.sum(v_order_rec)
        for i in range(len(pi)):
            pi[i].append(v_order_rec[i])
            pi[i] = tuple(pi[i])

        # record the data
        data_record.add(self.game.get_gameboard(), pi, self.q, self.player)

        # make the game move
        self.game.step(self.player,max_child.action[0],
                       max_child.action[1],max_child.action[2],
                       max_child.action[3],max_child.action[4])
        
        # update the node
        self.game.pprint()
        print("move action: ", max_child.action)
        self.player = max_child.player
        self.n = max_child.n # add 1 when backp
        self.w = max_child.w # add v when backp
        self.q = max_child.q # 1 / self.n * backp v_acc 
        self.p = max_child.p # from parent nn
        self.action = max_child.action
        self.children = max_child.children 
        self.meta_value = max_child.meta_value


def main():
    game = Game()
    mcts = MCTS(prob=1, action=(0,0,0,0,False), game=game, config=MctsConfig(), player=0)
    for _ in range(150):
        mcts.run_simu(200)
        mcts.move()

if __name__ == "__main__":
    main()


