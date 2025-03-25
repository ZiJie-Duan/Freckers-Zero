import torch
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


class Conv3DStack(nn.Module):
    def __init__(self):
        super(Conv3DStack, self).__init__()
        # 公共特征提取部分
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.residual_block1 = ResidualBlock(16, 16)
        #self.residual_block2 = ResidualBlock(16, 16)
        
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


def train(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    mask_loss = MaskLoss()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for gameboard, action_prob, value in train_loader:
            gameboard = gameboard.to(device)
            action_prob = action_prob.to(device)
            value = value.to(device)

            optimizer.zero_grad()
            p_action_prob, p_value = model(gameboard)
            # 计算双损失
            loss_img = mask_loss(p_action_prob, action_prob)
            loss_prob = F.mse_loss(p_value.view(-1), value)
            total_loss = loss_img + loss_prob
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {running_loss/len(train_loader):.8f}")



def train_bk(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for gameboard, action_prob, value in train_loader:
            gameboard = gameboard.to(device)
            action_prob = action_prob.to(device)
            value = value.to(device)

            optimizer.zero_grad()
            p_action_prob, p_value = model(gameboard)
            # 计算双损失
            loss_img = F.mse_loss(p_action_prob, action_prob)

            loss_prob = F.mse_loss(p_value.view(-1), value)
            total_loss = loss_img + loss_prob
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {running_loss/len(train_loader):.8f}")


