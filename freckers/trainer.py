import torch
import torch.optim as optim
import torch.nn.functional as F
from model import MaskLoss
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

class Trainer:

    def __init__(self, model, train_dataset, val_dataset, modelPath) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.modelPath = modelPath

    def get_dataloader(self, config):
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=config.batch_size, 
            shuffle=config.shuffle, num_workers=config.num_workers, 
            persistent_workers=True)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=config.batch_size, 
            shuffle=config.shuffle, num_workers=config.num_workers, 
            persistent_workers=True)

    def train(self, config):
        
        self.get_dataloader(config)

        num_epochs = config.epochs
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), 
                               lr=config.max_l_rate,
                               weight_decay=0.01)
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
                gameboard = gameboard.to(device)
                action_prob = action_prob.to(device)
                value = value.to(device)

                optimizer.zero_grad()
                p_action_prob, p_value = self.model(gameboard)
                # 计算双损失
                loss_img = mask_loss(p_action_prob, action_prob)
                loss_prob = F.mse_loss(p_value.view(-1), value)
                total_loss = loss_img + loss_prob
                
                total_loss.backward()
                optimizer.step()
                #scheduler.step()
                train_loss += total_loss.item()
            
            train_loss /= len(self.train_loader)
            train_loss_record.append(train_loss)

            self.model.eval()
            eval_loss = 0.0
            with torch.no_grad():
                for gameboard, action_prob, value in self.val_loader:
                    gameboard = gameboard.to(device)
                    action_prob = action_prob.to(device)
                    value = value.to(device)

                    p_action_prob, p_value = self.model(gameboard)
                    loss_img = mask_loss(p_action_prob, action_prob)
                    loss_prob = F.mse_loss(p_value.view(-1), value)
                    total_loss = loss_img + loss_prob
                    
                    eval_loss += total_loss.item()
            
            eval_loss /= len(self.val_loader)
            eval_loss_record.append(eval_loss)

            print(f"Epoch {epoch+1}, Training Loss: {train_loss:.8f}, Validation Loss: {eval_loss:.8f}")

        torch.save(self.model, self.modelPath)
