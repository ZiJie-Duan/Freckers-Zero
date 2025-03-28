from math import sqrt
from freckers_gym import RSTK
from deep_frecker import DeepFrecker
import numpy as np
import copy

class MCTS:
    def __init__(self, prob, action, config, 
                 deepfrecker0, deepfrecker1, player=0, root=False) -> None:
        
        self.deepfrecker0 = deepfrecker0
        self.deepfrecker1 = deepfrecker1
        self.player = player # 0/1
        self.config = config
        self.action = action # (r,c,nr,nc,grow)
        self.n = 0 # add 1 when backp
        self.w = 0 # add v when backp
        self.q = 0 # 1 / self.n * backp v_acc 
        self.p = prob # from parent nn

        self.children = [] 
        self.meta_value = 0
        self.root = root

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
        # temp_gb = game_temp.get_gameboard_matrix(self.player)
        # if (np.array_equal(temp_gb[0], temp_gb[3]) and
        #     np.array_equal(temp_gb[0], temp_gb[6]) and
        #     np.array_equal(temp_gb[0], temp_gb[9]) and
        #     np.array_equal(temp_gb[0], temp_gb[12])):
        if False:
            temp_gb = game_temp.get_gameboard_matrix(self.player)
            if (
                np.array_equal(temp_gb[6], temp_gb[9]) and
                np.array_equal(temp_gb[9], temp_gb[12]) and self.player == 0
                and max_child.action == (0, 0, 0, 0, True)):
                    print("\n\ndebug_rec Player:", self.player)
                    game_temp.pprint(debug = True)
                    for i in range(len(debug_rec)):
                        print(debug_rec[i])
                    print("max_child:", max_child.action)
                    print("max_child.n:", max_child.n)
                    print("max_child.w:", max_child.w)
                    print("max_child.q:", max_child.q)
                    print("max_child.p:", max_child.p)
                    print("max:", max)
                    print("self.temp_counter:", self.temp_counter)
                    input("Press Enter to continue...")
        #-----------------------------------------------------
        
        return max_child


    def add_dirichlet_noise(self):
        dirichlet_noise = np.random.dirichlet([self.config.dirichlet_alpha] * len(self.children))
        for i, child in enumerate(self.children):
            child.p = child.p * (1 - self.config.dirichlet_epsilon) + dirichlet_noise[i] * self.config.dirichlet_epsilon


    def expand(self, action_prob, rstk, game):

        for actions in rstk.get_action_space(self.player):
            base_loc = actions[0] # the location of the chess
            for action in actions[1:]: # loop to get the location where the chess will be placed
                self.children.append(
                    MCTS(
                        prob = action_prob[DeepFrecker.loc_trans(action[0], action[1])][base_loc[0]][base_loc[1]],
                        action = (base_loc[0], base_loc[1], action[0], action[1], False),
                        config = self.config,
                        player = 0 if self.player == 1 else 1,
                        deepfrecker0= self.deepfrecker0,
                        deepfrecker1= self.deepfrecker1
                        )
                )

        # check if the last two action is grow
        gamematrix = game.get_gameboard_matrix(self.player)
        skip_grow = False
        if self.player == 0:
            if (np.array_equal(gamematrix[6], gamematrix[9]) and
                np.array_equal(gamematrix[9], gamematrix[12])):
                skip_grow = True
        else:
            if (np.array_equal(gamematrix[7], gamematrix[10]) and
                np.array_equal(gamematrix[10], gamematrix[13])):
                skip_grow = True

        if not skip_grow:
            self.children.append(
                MCTS(
                    prob = action_prob[5][5][5], # fix the location of grow probability
                    action = (0, 0, 0, 0, True), # grow
                    config = self.config,
                    player = 0 if self.player == 1 else 1,
                    deepfrecker0= self.deepfrecker0,
                    deepfrecker1= self.deepfrecker1
                )
        )
        if self.root:
            self.add_dirichlet_noise()

    def simu(self, game):
        self.temp_counter = 1 if not hasattr(self, 'temp_counter') else self.temp_counter + 1
        deep_frecker = self.deepfrecker0 if self.player == 0 else self.deepfrecker1

        if self.n == 0:
            # eval
            action_prob, value = deep_frecker.run(
                game.get_gameboard_matrix(self.player))
            rstk = RSTK(game.get_gameboard())

            # expand
            self.expand(action_prob, rstk, game)
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
    

    def getPi(self): 
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
        
        return pi
    

    def cutMove(self, pi):  
        max_i = 0
        max_v = -99999
        for i, p in enumerate(pi):
            if p[-1] > max_v:
                max_v = p[-1]
                max_i = i

        max_child = self.children[max_i]
        
        # update the node
        self.player = max_child.player
        self.n = max_child.n # add 1 when backp
        self.w = max_child.w # add v when backp
        self.q = max_child.q # 1 / self.n * backp v_acc 
        self.p = max_child.p # from parent nn
        self.action = max_child.action
        self.children = max_child.children 
        self.meta_value = max_child.meta_value
        self.root = True

        # add dirichlet noise when the tree change the node
        self.add_dirichlet_noise()


class MCTSAgent:
    
    def __init__(self, deepfrecker0, deepfrecker1, mcts_config, first_player):
        self.deepfrecker0 = deepfrecker0
        self.deepfrecker1 = deepfrecker1
        self.config = mcts_config
        self.first_player = first_player
        self.mcts = MCTS(
            prob= 1,
            action= (0,0,0,0,False),
            config = mcts_config,
            deepfrecker0= self.deepfrecker0,
            deepfrecker1= self.deepfrecker1,
            player= first_player,
            root= True
        )
        self.rounds = mcts_config.search_step
    
    def reset(self):
        self.mcts = MCTS(
            prob= 1,
            action= (0,0,0,0,False),
            config = self.config,
            deepfrecker0= self.deepfrecker0,
            deepfrecker1= self.deepfrecker1,
            player= self.first_player,
            root= True
        )
        
    def simulate(self, game):
        for i in range(self.rounds):
            game_copy = copy.deepcopy(game)
            self.mcts.simu(game_copy)

    def getPi(self):
        return self.mcts.getPi() 

    def getAction(self,pi):
        max_i = 0
        max_v = -99999
        for i, p in enumerate(pi):
            if p[-1] > max_v:
                max_v = p[-1]
                max_i = i

        return pi[max_i]
    
    def cutMove(self, pi):
        self.mcts.cutMove(pi)

    def getPlayer(self):
        return self.mcts.player

