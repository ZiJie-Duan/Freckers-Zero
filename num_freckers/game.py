import numpy as np

class game:

    def __init__(self):
        self.gamebaord = np.zeros((13,8,8),np.int8) # red * 6 + blue * 6 + leaf
    
    def step(self,)