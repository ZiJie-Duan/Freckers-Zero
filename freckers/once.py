from deep_frecker import DeepFrecker, FreckerDataSet
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from fnet import MaskLoss, Conv3DStack
from mcts import mcts_data_collect


model = Conv3DStack()
mcts_data_collect(model, "N", f"C:\\Users\\lucyc\\Desktop\\freckers_data\\batch_1.h5", 30, 500)