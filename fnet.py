import torch
import torch.nn as nn
import torch.optim as optim

class Conv3DStack(nn.Module):
    def __init__(self):
        super(Conv3DStack, self).__init__()
        # 公共特征提取部分
        self.conv1 = nn.Conv2d(8, 256, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        # 图像输出头
        self.img_head = nn.Conv2d(256, 6, kernel_size=3, padding=1)

        # 数值概率输出头
        self.prob_conv = nn.Conv2d(256, 3, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 8 * 8, 64)
        self.relu_fc = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        
        # 公共特征处理
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))

        # 图像输出分支
        img_out = self.img_head(x)

        # 数值概率分支
        prob = self.relu4(self.prob_conv(x))
        prob = self.flatten(prob)
        prob = self.relu_fc(self.fc1(prob))
        prob = self.fc2(prob)
        prob_out = self.softmax(prob)

        return img_out, prob_out

def train(model, train_loader, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 双损失函数：图像用MSE，概率用交叉熵
    criterion_img = nn.MSELoss()
    criterion_prob = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets_img, targets_prob in train_loader:
            inputs = inputs.to(device)
            targets_img = targets_img.to(device)
            targets_prob = targets_prob.to(device)

            optimizer.zero_grad()
            outputs_img, outputs_prob = model(inputs)

            # 计算双损失
            loss_img = criterion_img(outputs_img, targets_img)
            loss_prob = criterion_prob(outputs_prob, targets_prob)
            total_loss = loss_img + loss_prob
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        print(f"Epoch {epoch+1}, Total Loss: {running_loss/len(train_loader):.4f}")

def inference(model, input_data):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_data).float()
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
        img_output, prob_output = model(input_tensor)
        return img_output.numpy(), prob_output.numpy()