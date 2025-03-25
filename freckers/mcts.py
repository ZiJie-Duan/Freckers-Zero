from math import sqrt
from freckers_gym import RSTK
from deep_frecker import DeepFrecker
from deep_frecker import DataRecord
import numpy as np
import copy
from game import Game
import os

class MCTS:
    def __init__(self, prob, action, config, deep_frecker, 
                 data_record=None, game=None, player=0) -> None:
        self.deep_frecker = deep_frecker
        self.data_record = data_record
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

    def select(self, game_temp):
        max = -999
        max_child = None

        # U(s,a) = C_puct * P(s,a) * Sqrt(Sum(N(s,b)_b_for_all_children)) / (1 + N(s,a)) 
        # all_num = Sqrt(Sum(N(s,b)_b_for_all_children))
        # all_num = sqrt(sum([c.n for c in self.children]))
        # all_num = Sqrt(Parents.N)
        all_num = sqrt(self.n)

        debug_rec = []

        for child in self.children:

            # puct = Q(s,a) + U(s,a)
            u = self.config.c * child.p * (all_num) / (1 + child.n)
            puct = child.q + u
            if max < puct:
                max = puct
                max_child = child
            
            debug_rec.append((puct, child.action, child.n, child.w, child.q, child.p))
        
        #-----------------------------------------------------
        if self.config.visulze and self.game != None:
            print("\n\ndebug_rec Player:", self.player)
            game_temp.pprint(debug = True)
            for i in range(len(debug_rec)):
                print(debug_rec[i])
            input("Press Enter to continue...")
        #-----------------------------------------------------
        
        return max_child


    def add_dirichlet_noise(self):
        dirichlet_noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(self.children))
        for i, child in enumerate(self.children):
            child.p = child.p * (1 - self.config.dirichlet_epsilon) + dirichlet_noise[i] * self.config.dirichlet_epsilon


    def expand(self, action_prob, rstk):

        for actions in rstk.get_action_space(self.player):
            base_loc = actions[0] # the location of the chess
            for action in actions[1:]: # loop to get the location where the chess will be placed
                
                self.children.append(
                    MCTS(
                        prob = action_prob[DeepFrecker.loc_trans(action[0], action[1])][base_loc[0]][base_loc[1]],
                        action = (base_loc[0], base_loc[1], action[0], action[1], False),
                        config = self.config,
                        player = 0 if self.player == 1 else 1,
                        deep_frecker=self.deep_frecker
                        )
                )

        self.children.append(
            MCTS(
                prob = action_prob[5][5][5], # fix the location of grow probability
                action = (0, 0, 0, 0, True), # grow
                config = self.config,
                player = 0 if self.player == 1 else 1,
                deep_frecker=self.deep_frecker
                )
        )

        # add dirichlet noise when the game start the first step
        if self.game != None:
            self.add_dirichlet_noise()

    def simu(self, game):

        if self.n == 0:
            # eval
            action_prob, value = self.deep_frecker.run(
                game.get_gameboard_matrix(self.player), self.player)
            rstk = RSTK(game.get_gameboard())
            # expand
            self.expand(action_prob, rstk)
            # backp
            self.meta_value = value[0] # value has two value, first is win, second is lose
            self.n += 1
            self.w += value[0]
            self.q = value[0] / self.n
            return value[0] # return estimated value and, but no reward

        else:
            # selection
            child = self.select(game)
            # move
            s,r,sn,end = game.step(self.player, *child.action)

            if end:
                # update the child node!!!
                child.n += 1
                child.w += r
                child.q = child.w / child.n

                self.n += 1
                self.w += r
                self.q = self.w / self.n
                return r # if end, return the reward instead of the estimated value
            else:
                # go deeper
                value = child.simu(game)
                # inverse value and r (i am not sure)
                # value = -1 * value

                # backp
                self.n += 1
                self.w += value
                self.q = self.w / self.n

            return value # if not end, return the estimated value and the child's reward
    

    def run_simu(self, rounds):
        for i in range(rounds):
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
        self.data_record.add(self.game.get_gameboard_matrix(self.player), pi, 0, self.player)
        # here we pass a temp value 0, because we want to update the value later
        # the value depends on the end of the game

        # make the game move
        s,r,sn,end = self.game.step(self.player,max_child.action[0],
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

        # add dirichlet noise when the tree change the node
        self.add_dirichlet_noise()

        if end:
            if r == 0:
                print("drop the data")
                self.data_record.drop()
            else:
                self.data_record.update_value_and_save(r)

        winner = None
        if r == 1:
            winner = 0 if self.player == 1 else 1
        return end, winner


def mcts_data_collect(model, thread_num, file, config, rounds=100, sim_step=300, model2=None):
    deep_frecker = DeepFrecker(model, model2)
    data_record = DataRecord(file=file)

    for j in range(rounds):

        game = Game()
        mcts = MCTS(prob=2, action=(0,0,0,0,False), 
                    game=game, config=config, player=1,
                    deep_frecker=deep_frecker, data_record=data_record)

        for i in range(300):
            print("线程", thread_num, "第", j, "轮游戏 ", "第", i, "步 模拟进行中")
            if i > 30:
                mcts.config.t = 0.2
            elif i > 60:
                mcts.config.t = 0.01
            else:
                mcts.config.t = 1
            # if i > 100:
            #     mcts.config.visulze = True
            # else:
            #     mcts.config.visulze = False
            mcts.run_simu(sim_step)
            end, _ = mcts.move()
            if end:
                break


def mcts_competition(model, thread_num, file, config, rounds=100, sim_step=300, model2=None):
    deep_frecker = DeepFrecker(model, model2)
    data_record = DataRecord(file=file)
    winner_record = []

    for j in range(rounds):

        game = Game()
        mcts = MCTS(prob=2, action=(0,0,0,0,False), 
                    game=game, config=config, player=1,
                    deep_frecker=deep_frecker, data_record=data_record)

        for i in range(300):
            print("线程", thread_num, "第", j, "轮游戏 ", "第", i, "步 模拟进行中")
            if i > 30:
                mcts.config.t = 0.2
            elif i > 60:
                mcts.config.t = 0.01
            else:
                mcts.config.t = 1
            # if i > 100:
            #     mcts.config.visulze = True
            # else:
            #     mcts.config.visulze = False
            mcts.run_simu(sim_step)
            end, winner = mcts.move()
            if end:
                winner_record.append(winner)
                break
    
    return winner_record

