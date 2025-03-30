
import numpy as np
import h5py

class DataRecord:
    def __init__(self, file):
        print("file: ", file)
        self.file = file
        self.count = 0
        self.write_first = True

        # 初始化三个独立数组容器
        self.memo_gameboard = np.empty((0, 16, 8, 8), dtype=np.float32)
        self.memo_action_prob = np.empty((0, 65, 8, 8), dtype=np.float32)
        self.memo_value = np.empty((0,), dtype=np.float32)
        
    def rotate_point180(self, x, y):
        return 8 - x, 8 - y


    def add(self, gameboard, action_prob, value):
        """
        gameboard: 3x8x8
        action_prob: [(c,r,rn,cn,v), (c,r,rn,cn,v), ...]
        # grow always in the last element in action space
        value: 1
        """
        action_prob_m = np.zeros((65, 8, 8), dtype=np.float32)
        for i in range(len(action_prob) - 1): # the last one is the grow probability
            action_prob_m[action_prob[i][2]*8 + action_prob[i][3],
                          action_prob[i][0], 
                          action_prob[i][1]]\
                        = action_prob[i][4]
            
            # add the grow probability
            # 在每一个合法动作65层中，添加grow概率
            action_prob_m[64, action_prob[i][0], action_prob[i][1]] = action_prob[-1][4]

        gameboard = np.array([gameboard], dtype=np.float32)
        action_prob_m = np.array([action_prob_m], dtype=np.float32)
        value = np.array([value], dtype=np.float32)

        self.memo_gameboard = np.concatenate((self.memo_gameboard, gameboard))
        self.memo_action_prob = np.concatenate((self.memo_action_prob, action_prob_m))
        self.memo_value = np.concatenate((self.memo_value, value))
        self.count += 1
    

    def update_value_and_save(self, value):
        p = -1 * value if len(self.memo_value) % 2 == 0 else value
        for i in range(len(self.memo_value)):
            self.memo_value[i] = p
            p = -1 * p

        self.save()


    def drop(self):
        # 清空缓存
        self.memo_gameboard = np.empty((0, 16, 8, 8), dtype=np.int32)
        self.memo_action_prob = np.empty((0, 65, 8, 8), dtype=np.int32)
        self.memo_value = np.empty((0,), dtype=np.int32)


    def save(self, file=None):
        if file == None:
            file = self.file

        with h5py.File(file, 'a') as f:  # 始终使用追加模式
            # 初始化数据集（如果不存在）
            if 'gameboard' not in f:
                f.create_dataset(
                    'gameboard',
                    data=self.memo_gameboard,
                    maxshape=(None, 16, 8, 8),
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
        self.memo_gameboard = np.empty((0, 16, 8, 8), dtype=np.int32)
        self.memo_action_prob = np.empty((0, 65, 8, 8), dtype=np.int32)
        self.memo_value = np.empty((0,), dtype=np.int32)


