import numpy as np
from scipy.signal import convolve2d
from game import Game

game = Game()
game.pprint()
game.step(0, 0, 1, 0, 0, grow=False)
game.pprint()
game.step(1, 6, 1, 7, 1, grow=False)
game.pprint()
game.step(0, 0, 5, 0, 7, grow=False)
game.pprint()

