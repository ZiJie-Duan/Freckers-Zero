import torch
import torch.optim as optim
import torch.nn.functional as F
from model import MaskLoss, FreckersNet
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:

    def __init__(self, checkpointFile, config, no_checkpoint= False) -> None:
        self.checkpointFile = checkpointFile
        self.config = config
        
        self.model = None
        self.optimizer = None
        self.loss_rec = []
        self.epoch_ct = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if no_checkpoint:
            print("[Trainer]: No Check Point")
            self.model = FreckersNet()
            self.model = self.model.to(self.device)
            self.optimizer = optim.Adam(self.model.parameters(), 
                            lr=self.config.max_l_rate,
                            weight_decay=0.01)
            self.epoch_ct = 0
        else:
            print("[Trainer]: Use Check Point")
            self.recover()
 
    def recover(self):
        # 加载检查点
        checkpoint = torch.load(self.checkpointFile)

        # 必须按原参数重新初始化优化器
        model = FreckersNet()
        model = model.to(self.device)  # 需要与原模型结构一致
        optimizer = optim.Adam(model.parameters(), 
                            lr=self.config.max_l_rate,
                            weight_decay=0.01)

        # 关键步骤：加载保存的状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # 恢复其他训练状态
        self.epoch_ct = checkpoint['epoch']
        self.loss_rec = checkpoint["loss_rec"]

        self.model = model
        self.optimizer = optimizer

    def get_dataloader(self, config, train_dataset, val_dataset):
        self.train_loader = DataLoader(
            train_dataset, batch_size=config.batch_size, 
            shuffle=config.shuffle)
        self.val_loader = DataLoader(
            val_dataset, batch_size=config.batch_size, 
            shuffle=config.shuffle)


    def train(self, config, checkPointFile, train_dataset, val_dataset):
        
        self.get_dataloader(config, train_dataset, val_dataset)

        num_epochs = config.epochs

        # scheduler = OneCycleLR(
        #     optimizer,
        #     max_lr=config.max_l_rate,  # 最大学习率
        #     steps_per_epoch=len(self.train_loader),
        #     epochs=num_epochs,
        #     pct_start=0.4  # warmup 的比例
        # )
        mask_loss = MaskLoss()

        train_loss_record = []
        eval_loss_record = []

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for gameboard, action_prob, value in self.train_loader:
                gameboard = gameboard.to(self.device)
                action_prob = action_prob.to(self.device)
                value = value.to(self.device)

                self.optimizer.zero_grad()
                p_action_prob, p_value = self.model(gameboard)
                # 计算双损失
                loss_img = mask_loss(p_action_prob, action_prob)
                loss_prob = F.mse_loss(p_value.view(-1), value)
                total_loss = loss_img + loss_prob
                
                total_loss.backward()
                self.optimizer.step()
                #scheduler.step()
                train_loss += total_loss.item()
            
            train_loss /= len(self.train_loader)
            train_loss_record.append(train_loss)
            self.loss_rec.append(train_loss)

            # self.model.eval()
            # eval_loss = 0.0
            # with torch.no_grad():
            #     for gameboard, action_prob, value in self.val_loader:
            #         gameboard = gameboard.to(self.device)
            #         action_prob = action_prob.to(self.device)
            #         value = value.to(self.device)

            #         p_action_prob, p_value = self.model(gameboard)
            #         loss_img = mask_loss(p_action_prob, action_prob)
            #         loss_prob = F.mse_loss(p_value.view(-1), value)
            #         total_loss = loss_img + loss_prob
                    
            #         eval_loss += total_loss.item()
            
            # eval_loss /= len(self.val_loader)
            # eval_loss_record.append(eval_loss)

            self.epoch_ct += 1
            print(f"Epoch {self.epoch_ct}, Training Loss: {train_loss:.8f}")

        checkpoint = {
            'epoch': self.epoch_ct,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "loss_rec" : self.loss_rec
        }

        torch.save(checkpoint, checkPointFile)
