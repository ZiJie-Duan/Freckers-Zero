import numpy as np
from scipy.signal import convolve2d
from game import Game
from freckers_gym import RSTK

game = Game()

green_layer = np.array([
    [1, 1, 1, 1, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 1],
    [0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0]
])

red_layer = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

blue_layer = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0]
])

gameboard = [red_layer, blue_layer,green_layer]
game.gamebaord = gameboard
game.pprint()
r = RSTK(gameboard)
print(r.get_action_space(1))


# Rounds:  42
# 🟢🟢🟢🟢⚪🟢🟢🔵
# 🟢🟢🟢🟢🟢🟢🟢⚪
# 🟢🔴🔴🔴🔴🟢🟢🟢
# 🟢🟢🟢🟢🟢🔴🔴🟢
# ⚪⚪⚪⚪🟢🟢🟢🟢
# ⚪⚪🔵⚪⚪⚪🔵🔵
# 🔵⚪⚪⚪⚪⚪🔵⚪
# ⚪🟢🟢🟢🟢🟢🟢⚪
# move action:  (6, 7, 0, 7, False)