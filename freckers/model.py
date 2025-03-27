import torch
from torch.utils.data import Dataset
import h5py
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class FreckersNet(nn.Module):
    def __init__(self):
        super(FreckersNet, self).__init__()
        # 公共特征提取部分 
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.residual_block1 = ResidualBlock(16, 16)
        #self.residual_block2 = ResidualBlock(32, 32)
        
        # 图像输出头
        self.img_head = nn.Conv2d(16, 65, kernel_size=3, padding=1)
        self.relu_img = nn.ReLU()

        # 数值概率输出头
        self.prob_conv = nn.Conv2d(16, 5, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(5 * 8 * 8, 16)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        raw = x
        # 公共特征处理
        x = self.relu1(self.conv1(x))
        x = self.residual_block1(x)
        #x = self.residual_block2(x)

        # 图像输出分支
        img_out = self.img_head(x)
        img_out = self.relu_img(img_out)

        # 数值概率分支
        prob = self.relu4(self.prob_conv(x))
        prob = self.flatten(prob)
        prob = self.relu_fc(self.fc1(prob))
        prob_out = torch.sigmoid(self.fc2(prob))

        return img_out, prob_out


class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()

    def forward(self, predictions, targets):
        musk = targets != 0.0
        loss = F.mse_loss(predictions[musk], targets[musk])
        return loss



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

