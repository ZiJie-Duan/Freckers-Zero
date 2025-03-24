from deep_frecker import DeepFrecker, FreckerDataSet
from torch.utils.data import DataLoader, random_split
import torch
import torch.optim as optim
import torch.nn.functional as F
from fnet import MaskLoss, Conv3DStack
from mcts import mcts_data_collect
import win32api
import win32process
import win32con

# 获取当前进程句柄
pid = win32api.GetCurrentProcessId()
handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, True, pid)
win32process.SetPriorityClass(handle, win32process.HIGH_PRIORITY_CLASS)

#model = torch.load(r"C:\Users\lucyc\Desktop\freckers_data\batch_2_no_inverse.pth", weights_only=False)
model = Conv3DStack()
mcts_data_collect(model, "N", f"C:\\Users\\lucyc\\Desktop\\freckers_data\\New_v1.h5", 100, 400)