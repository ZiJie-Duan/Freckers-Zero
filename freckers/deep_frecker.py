
from fnet import Conv3DStack
import torch
import numpy as np
import h5py

class DataRecord:
    def __init__(self, file = "data.h5", save_interval = 3000):
        self.file = file
        self.count = 0
        self.write_first = True
        self.save_interval = save_interval

        # 初始化三个独立数组容器
        self.memo_gameboard = np.empty((0, 3, 8, 8), dtype=np.float32)
        self.memo_action_prob = np.empty((0, 65, 8, 8), dtype=np.float32)
        self.memo_value = np.empty((0,), dtype=np.float32)
        
    def rotate_point180(self, x, y):
        return 8 - x, 8 - y

    def add(self, gameboard, action_prob, value, player):
        """
        gameboard: 3x8x8
        action_prob: [(c,r,rn,cn,v), (c,r,rn,cn,v), ...]
        value: 1
        """
        action_prob_m = np.zeros((65, 8, 8), dtype=np.float32)
        for i in range(len(action_prob)):
            action_prob_m[action_prob[i][2]*8 + action_prob[i][3],
                          action_prob[i][0], 
                          action_prob[i][1]] \
                        = action_prob[i][4]

        if player != 0:
            gameboard = np.rot90(gameboard, 2) # rotate 180 degrees
            gameboard[[0, 1]] = gameboard[[1, 0]] # swap red and blue
            action_prob_m = np.rot90(action_prob_m, 2)

        gameboard = np.array([gameboard], dtype=np.float32)
        action_prob_m = np.array([action_prob_m], dtype=np.float32)
        value = np.array([value], dtype=np.float32)
        
        self.memo_gameboard = np.concatenate((self.memo_gameboard, gameboard))
        self.memo_action_prob = np.concatenate((self.memo_action_prob, action_prob_m))
        self.memo_value = np.concatenate((self.memo_value, value))
        self.count += 1

        if self.count % self.save_interval == 0:
            self.save()


    def save(self):
        with h5py.File(self.file, 'a') as f:  # 始终使用追加模式
            # 初始化数据集（如果不存在）
            if 'gameboard' not in f:
                f.create_dataset(
                    'gameboard',
                    data=self.memo_gameboard,
                    maxshape=(None, 3, 8, 8),
                    chunks=True
                )
                f.create_dataset(
                    'action_prob',
                    data=self.memo_action_prob,
                    maxshape=(None, 65, 8, 8),
                    chunks=True
                )
                f.create_dataset(
                    'value',
                    data=self.memo_value,
                    maxshape=(None,),
                    chunks=True
                )
            else:
                # 扩展数据集
                for name, data in [('gameboard', self.memo_gameboard),
                                 ('action_prob', self.memo_action_prob),
                                 ('value', self.memo_value)]:
                    dataset = f[name]
                    old_size = dataset.shape[0]
                    new_size = old_size + data.shape[0]
                    
                    # 调整数据集大小
                    dataset.resize(new_size, axis=0)
                    
                    # 写入新数据
                    dataset[old_size:] = data
        
        # 清空缓存
        self.memo_gameboard = np.empty((0, 3, 8, 8), dtype=np.int32)
        self.memo_action_prob = np.empty((0, 65, 8, 8), dtype=np.int32)
        self.memo_value = np.empty((0,), dtype=np.int32)

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