
from fnet import Conv3DStack
import torch
import numpy as np
import h5py

class DataRecord:
    def __init__(self, file = "data.h5"):
        self.file = file
        self.memory = []
        self.count = 0
        self.write_first = True
        
    def rotate_point180(self, x, y):
        return 8 - x, 8 - y

    def add(self, gameboard, action_prob, value, player):
        """
        gameboard: 3x8x8
        action_prob: [(c,r,rn,cn,v), (c,r,rn,cn,v), ...]
        value: 1
        """
        action_prob_m = np.zeros((65,8,8))
        for i in range(len(action_prob)):
            action_prob_m[action_prob[i][2]*8 + action_prob[i][3],
                          action_prob[i][0], 
                          action_prob[i][1]] \
                        = action_prob[i][4]

        if player != 0:
            gameboard = np.rot90(gameboard, 2) # rotate 180 degrees
            gameboard[[0, 1]] = gameboard[[1, 0]] # swap red and blue
            action_prob_m = np.rot90(action_prob_m, 2)

        self.memory.append((gameboard, action_prob_m, value))
        self.count += 1

        if self.count % 3000 == 0:
            self.save()

    def save(self):
        if self.write_first:
            with h5py.File(self.file, 'w') as f:
                f.create_dataset('memory', data=self.memory)
            self.write_first = False
        else:
            with h5py.File(self.file, 'a') as f:
                f.create_dataset('memory', data=self.memory)

class DeepFrecker:
    def __init__(self):
        self.model = Conv3DStack()
    
    def run(self, gameboard, player):
        # 待验证
        if player == 0:
            input_data = gameboard
        else:
            input_data = np.rot90(gameboard, 2) # rotate 180 degrees
            input_data[[0, 1]] = input_data[[1, 0]] # swap red and blue

        action_prob, value = self.inference(input_data)

        if player == 0:
            return action_prob[0], value[0]
        else:
            return np.rot90(action_prob, 2)[0], value[0]

    
    def inference(self, input_data):
        self.model.eval()

        with torch.no_grad():
            input_tensor = torch.tensor(input_data.copy()).float()
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            img_output, prob_output = self.model(input_tensor)
            return img_output.numpy(), prob_output.numpy()


    def loc_trans(row, col):
        return (row*8 + col) 