import numpy as np
from scipy.signal import convolve2d

class Game:

    def __init__(self, rounds_limit = 250):
        self.gameboard_memory = [np.zeros((3,8,8),np.int8) for _ in range(4)]
        self.gamebaord = np.zeros((3,8,8),np.int8) # [red, blue, leaf]
        self.gamebaord[0][0] = np.array([0,1,1,1,1,1,1,0]) # red
        self.gamebaord[1][7] = np.array([0,1,1,1,1,1,1,0]) # blue
        self.gamebaord[2][0] = np.array([1,0,0,0,0,0,0,1]) # leaf
        self.gamebaord[2][1] = np.array([0,1,1,1,1,1,1,0]) # leaf
        self.gamebaord[2][6] = np.array([0,1,1,1,1,1,1,0]) # leaf
        self.gamebaord[2][7] = np.array([1,0,0,0,0,0,0,1]) # leaf
        self.red = 0
        self.blue = 1
        self.leaf = 2
        self.top_row = 0
        self.bottom_row = 7
        self.rounds = 0
        self.rounds_limit = rounds_limit

        self.gameboard_memory.append(self.gamebaord.copy())
    
    def custom_reward_gravity(self, player, r, c, rn, cn, grow):
        reward = 0
        if player == 0:
            reward += (rn - r)**2 * 0.03
        elif player == 1:
            reward += (rn - r)**2 * 0.03
        return reward

    def init(self):
        self.__init__(self.rounds_limit)

    def win_check(self):
        if np.sum(self.gamebaord[self.red][self.bottom_row]) == 6:
            return self.red
        elif np.sum(self.gamebaord[self.blue][self.top_row]) == 6:
            return self.blue
        else:
            return None

    def grow(self, player):
        """
        player: 0, 1
        grow Leaf
        """
        if player == 0:
            layer = self.gamebaord[self.red]
        else:
            layer = self.gamebaord[self.blue]

        # get all the leaf space
        kernel = np.ones((3, 3), dtype=np.uint8)
        convolved = convolve2d(layer, kernel, mode='same', boundary='fill', fillvalue=0)
        self.gamebaord[self.leaf] = np.where(convolved > 0, 1, self.gamebaord[self.leaf])

        # remove the leaf where fogs stands 
        all_fogs = np.nonzero(self.gamebaord[0] + self.gamebaord[1])
        self.gamebaord[self.leaf][all_fogs] = 0

    def step(self, player, r, c, rn, cn, grow = False):
        """
        return (S, R, S', done)
        """
        self.rounds += 1

        if grow:
            self.grow(player)
        else:   
            self.gamebaord[player][r][c] = 0
            self.gamebaord[player][rn][cn] = 1
            self.gamebaord[self.leaf][rn][cn] = 0
        
        winner = self.win_check()
        reward = 0 if winner == None else 1 if winner == player else -1
        reward += self.custom_reward_gravity(player, r, c, rn, cn, grow)
        done = False if (self.rounds < self.rounds_limit) and (winner == None) else True

        musk_s = None
        musk_sn = None
        if player == 0:
            musk_s = np.zeros([1,8,8])
            musk_sn = np.ones([1,8,8])
        else: 
            musk_sn = np.zeros([1,8,8])
            musk_s = np.ones([1,8,8])

        s = np.concatenate(self.gameboard_memory + [musk_s.copy()], axis=0)
    
        self.gameboard_memory.pop(0)
        self.gameboard_memory.append(self.gamebaord.copy())

        sn = np.concatenate(self.gameboard_memory + [musk_sn.copy()], axis=0)

        return (
            s,
            reward,
            sn,
            done
        )

    def get_gameboard_matrix(self, player):
        musk = np.ones((1, 8, 8), dtype=np.int8)
        if player == 0:
            musk = np.zeros((1, 8, 8), dtype=np.int8)
            
        return np.concatenate(self.gameboard_memory + [musk.copy()], axis=0)

    def get_gameboard(self):
        return self.gamebaord.copy()


    def pprint(self, debug = False):
        if not debug:
            print("\nRounds: ", self.rounds)
        for r in range(8):
            for c in range(8):
                if self.gamebaord[0][r][c] == 1:
                    print('🔴', end='')  # RedCircle
                elif self.gamebaord[1][r][c] == 1:
                    print('🔵', end='')  # BlueCircle
                elif self.gamebaord[2][r][c] == 1:
                    print('🟢', end='')  # GreenCircle
                else:
                    print('⚪', end='')  # white circle
            print()
            
# g = Game()
# g.grow(1)
# g.step(1,7,2,6,0,False)
# g.grow(1)
# g.action_space(1)