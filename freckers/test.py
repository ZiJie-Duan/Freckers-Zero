import numpy as np
from scipy.signal import convolve2d
from game import Game
from freckers_gym import RSTK
from mcts import MCTS
from fnet import Conv3DStack
from deep_frecker import DeepFrecker
from deep_frecker import DataRecord
import copy

class MCTSTest(MCTS):
    def __init__(self, prob, action, config, deep_frecker, 
                 data_record=None, game=None, player=0) -> None:
        super().__init__(prob, action, config, deep_frecker, 
                         data_record, game, player)
        

    def get_pi(self):
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

    def show_pi(self):
        t_v = sum([c.n**(1/self.config.t) for c in self.children])
        print("sum child n: ", t_v)
        print("self n: ", self.n)
        pi = []

        max = -999
        max_child = 0
        v_order_rec = [] # get the probability of each action

        for i, child in enumerate(self.children):
            print("child action: ", child.action, "child n: ", child.n)
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
    

def convert_emoji_board_to_gameboard(emoji_board):
    """
    将表情符号棋盘转换为三个numpy矩阵层（红色、蓝色和绿色）
    
    参数:
        emoji_board: 包含表情符号的字符串，表示棋盘状态
        
    返回:
        green_layer, red_layer, blue_layer: 三个numpy矩阵
    """
    # 分割成行
    rows = emoji_board.strip().split('\n')
    
    # 初始化三个8x8的矩阵
    green_layer = np.zeros((8, 8), dtype=int)
    red_layer = np.zeros((8, 8), dtype=int)
    blue_layer = np.zeros((8, 8), dtype=int)
    
    # 遍历每个表情符号并填充相应的矩阵
    for i, row in enumerate(rows):
        for j, emoji in enumerate(row):
            if emoji == '🟢':  # 绿色
                green_layer[i, j] = 1
            elif emoji == '🔴':  # 红色
                red_layer[i, j] = 1
            elif emoji == '🔵':  # 蓝色
                blue_layer[i, j] = 1
            # 白色（空）不需要设置，因为矩阵已初始化为0
    
    return np.array([red_layer, blue_layer, green_layer])

# 示例棋盘
emoji_board = """
🟢🟢🟢🟢🟢🟢🟢🟢
🟢🟢🟢🟢🟢🔵🟢⚪
🟢🟢🔵🟢🟢🟢🟢🟢
🟢🟢🔵🟢🟢🟢🔵🟢
🟢🟢🟢🟢🟢🟢🟢🟢
🟢🔵⚪🟢🔵🟢⚪🟢
🟢🔴🟢🟢🟢🟢⚪🟢
🔴🔴🟢🔴⚪🔴🔴🟢
"""

# 转换棋盘
game = Game()
game.gamebaord = convert_emoji_board_to_gameboard(emoji_board)
game.pprint()

r = RSTK(game.get_gameboard())
print(r.get_action_space(0))

class MctsConfig:
    def __init__(self) -> None:
        self.c = 2
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

        self.dirichlet_alpha = 0.03
        self.dirichlet_epsilon = 0.25

data_record = DataRecord(file="test.h5")
model = Conv3DStack()
mtcs = MCTSTest(1, (0,0,0,0,False), MctsConfig(), DeepFrecker(), data_record, game=game, player=0)

for i in range(300):
    mtcs.config.counter = i
    if i > 300:
        mtcs.config.visulze = True
    game = copy.deepcopy(mtcs.game)
    mtcs.simu(game)

print(mtcs.show_pi())
mtcs.move()