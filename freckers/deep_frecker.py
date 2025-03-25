
from fnet import Conv3DStack, train
import torch
import numpy as np
import h5py
from torch.utils.data import Dataset
from torch.utils.data import DataLoader



class FreckerDataSet(Dataset):
    def __init__(self, file_path):
        """
        初始化H5Dataset对象。

        参数:
        - file_path: HDF5文件的路径
        - dataset_name: HDF5文件中数据集的名称
        """
        self.file_path = file_path
        self.gameboard = None
        self.action_prob = None
        self.value = None

        with h5py.File(self.file_path, 'r') as file:
            self.gameboard = file['gameboard'][:]  # 将数据加载到内存中
            self.action_prob = file['action_prob'][:]  # 将数据加载到内存中
            self.value = file['value'][:]  # 将数据加载到内存中

    def __len__(self):
        """
        返回数据集的长度。
        """
        return len(self.gameboard)


    def __getitem__(self, idx):
        """
        根据索引返回数据集中的一个样本。

        参数:
        - idx: 样本的索引

        返回:
        - 样本数据
        """
        gameboard = torch.tensor(self.gameboard[idx], dtype=torch.float32)
        action_prob = torch.tensor(self.action_prob[idx], dtype=torch.float32)
        value = torch.tensor(self.value[idx], dtype=torch.float32)
        return gameboard, action_prob, value



class DataRecord:
    def __init__(self, file = "data.h5"):
        self.file = file
        self.count = 0
        self.write_first = True

        # 初始化三个独立数组容器
        self.memo_gameboard = np.empty((0, 16, 8, 8), dtype=np.float32)
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
                          action_prob[i][1]]\
                        = action_prob[i][4]

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



class DeepFrecker:
    def __init__(self, model, model2=None):
        self.model = model
        self.model2 = model2

    def run(self, gameboard, player):
        model = None
        if self.model2 != None and player == 1:
            model = self.model2
        else:
            model = self.model

        action_prob, value = self.inference(gameboard, model)

        if player == 0:
            return action_prob[0], value[0]
        else:
            return np.rot90(action_prob, 2)[0], value[0]

    
    def inference(self, input_data, model):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        model.to(device)

        with torch.no_grad():
            input_tensor = torch.tensor(input_data.copy()).float()
            input_tensor = input_tensor.to(device)
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            
            img_output, prob_output = model(input_tensor)
            
            img_output = img_output.cpu().numpy()
            prob_output = prob_output.cpu().numpy()
            return img_output, prob_output


    def loc_trans(row, col):
        return (row*8 + col) 


    def train(self, file, epoch):

        # HDF5文件路径和数据集名称
        file_path = file

        # 创建H5Dataset对象
        dataset = FreckerDataSet(file_path)

        # 创建 DataLoader 对象
        batch_size = 128  # 每个批次的大小
        shuffle = True   # 是否打乱数据
        num_workers = 4  # 使用多少个子进程加载数据

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            persistent_workers=True 
        )

        train(self.model, dataloader, epoch)

    def save_model(self, file):
        torch.save(self.model, file)

    def load_model(self, file):
        self.model = torch.load(file)



# if __name__ == '__main__':
#     dataset = FreckerDataSet(r"C:\Users\lucyc\Desktop\freckers_zero\data.h5")
#     print(dataset.gameboard.shape)
#     print(dataset.action_prob.shape)
#     print(dataset.value.shape)

#     gameboard, action_prob, value = dataset[6595]
#     print(gameboard)
#     print(action_prob)
#     print(value)
