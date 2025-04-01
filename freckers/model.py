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
        self.conv1 = nn.Conv2d(16, 128, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.residual_block1 = ResidualBlock(128, 256)
        self.residual_block2 = ResidualBlock(256, 256)
        self.residual_block3 = ResidualBlock(256, 256)
        
        # 图像输出头
        self.img_head = nn.Conv2d(256, 65, kernel_size=3, padding=1)

        # 数值概率输出头
        self.prob_conv = nn.Conv2d(256, 5, kernel_size=3, padding=1)
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
        x = self.residual_block2(x)
        x = self.residual_block3(x)
        # x = self.residual_block3(x)
        # x = self.residual_block4(x)


        # 图像输出分支  
        img_out = self.img_head(x)

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

        # OK 这是一个非常诡异的损失函数
        # 我不知道，我觉得还是蛮有道理的 就是不确定
        # 在我的设计中 模型的行为概率矩阵是 B, 65, 8, 8
        # B 是batch size, 65 是64个合法动作+1个grow动作 （grow在第65个通道）
        # 8, 8 是游戏棋盘的大小
        # grow 是一个全局的动作，不需要指定位置
        # 这就是我要将所有合法位置的第65个通道求平均的原因
        # 我希望神经网络在想要grow的时候 尽可能的在所有合法位置上输出grow 信号
        # 我将 64个合法动作的概率 和 grow的概率 分开计算

        device = "cuda" if torch.cuda.is_available() else "cpu"
        predictions = predictions.to(device)
        targets = targets.to(device)

        # 首先将移动概率和 grow概率分开
        pred_main = predictions[:, :-1]  # (b, 64, 8, 8)
        pred_grow = predictions[:, -1]   # (b, 8, 8)
        label_main = targets[:, :-1]
        label_grow = targets[:, -1]

        # 创建非零的遮罩
        grow_mask = label_grow != 0
        label_grow = label_grow * grow_mask
        pred_grow = pred_grow * grow_mask

        # 然后计算 grow 的统计信息
        label_grow_counts = torch.count_nonzero(label_grow, dim=(1, 2))
        label_grow_sums = torch.sum(label_grow, dim=(1, 2))
        pred_grow_counts = torch.count_nonzero(pred_grow, dim=(1, 2))
        pred_grow_sums = torch.sum(pred_grow, dim=(1, 2))

        # 求出grow 的平均数值
        label_grow_mean = torch.zeros_like(label_grow_sums)
        mask = label_grow_counts != 0
        label_grow_mean[mask] = label_grow_sums[mask] / label_grow_counts[mask]

        pred_grow_mean = torch.zeros_like(pred_grow_sums)
        mask = pred_grow_counts != 0
        pred_grow_mean[mask] = pred_grow_sums[mask] / pred_grow_counts[mask]

        # 建立移动行为的遮罩
        main_mask = label_main != 0
        label_main = label_main * main_mask
        pred_main = pred_main * main_mask

        # 接下来，将整个移动行为概率矩阵拉平，并加入grow的平均数值
        B, C, H, W = pred_main.shape
        main_mask_flat = main_mask.view(B, -1)
        main_mask_flat = torch.concat((main_mask_flat, torch.ones(B, 1, dtype=torch.bool, device=device)), dim=1)
        pred_main_flat = pred_main.view(B, -1)  # -1 自动计算 H*W
        pred_main_flat = torch.concat((pred_main_flat, pred_grow_mean.view(B, 1)), dim=1)
        label_main_flat = label_main.view(B, -1)  # -1 自动计算 H*W
        label_main_flat = torch.concat((label_main_flat, label_grow_mean.view(B, 1)), dim=1)

        # 然后，用拉平的遮罩将所有的生效数值取出，并进行交叉熵计算
        loss = 0
        count = 0
        for i in range(B):
            log_softmax = F.log_softmax(pred_main_flat[i][main_mask_flat[i]])
            labels = label_main_flat[i][main_mask_flat[i]]
            loss += - (labels * log_softmax).sum()  # (B, 8, 8)
            count += len(labels)

        # 最后，返回平均损失
        return loss/count


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

