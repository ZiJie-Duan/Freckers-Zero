from global_config import MctsConfig, FreckersConfig
from deep_frecker import DeepFrecker
from data_record import DataRecord
from mcts_agent import MCTSAgent
from game import Game
from simulator import Simulator
from model import FreckersNet, FreckerDataSet
from trainer import Trainer
import torch
from torch.utils.data import DataLoader, random_split

class IterManager:

    def __init__(self) -> None:
        self.cfg = FreckersConfig()
        self.simulator = None
        self.trainer = None

    def simulation_init(self):
        # load the model
        model = None
        if self.cfg.iter_now == 0:
            model = FreckersNet()
        else:
            model = torch.load(
                self.cfg.model_base_dir 
                + "\\" + str(self.cfg.iter_now) + ".pth")
        
        deepfrecker = DeepFrecker(model=model)
        datarecorder = DataRecord(
            file=self.cfg.dataset_base_dir 
            + "\\" + str(self.cfg.iter_now + 1) + ".h5")
        
        mcts_agent = MCTSAgent(
            deepfrecker0=deepfrecker,
            deepfrecker1=deepfrecker,
            mcts_config= self.cfg.mcts_config,
            first_player=0
        )

        game = Game(self.cfg.game_rounds_limit)

        self.simulator = Simulator(
            game=game, mcts_agent=mcts_agent, dataRecorder=datarecorder
        )


    def load_dataset(self):
        now = self.cfg.iter_now
        datafiles = []
        if now == 0:
            datafiles.append(self.cfg.dataset_base_dir + "\\1" + ".h5")
        else:
            for i in range(
                max(1, now - self.cfg.training_dataset_cross),
                now + 1):

                datafiles.append(
                    self.cfg.dataset_base_dir + "\\" + str(i) + ".h5"
                )
        
        datasets = [FreckerDataSet(x) for x in datafiles]
        dataset = torch.utils.data.ConcatDataset(datasets)

        # 按照顺序分割数据集
        train_size = int(
            self.cfg.training_dataset_eval_rate * len(dataset))

        # 使用 Subset 分割数据集
        train_dataset = torch.utils.data.Subset(dataset, range(train_size))
        val_dataset = torch.utils.data.Subset(dataset, range(train_size, len(dataset)))

        select_rate = self.cfg.training_dataset_select_rate
        train_dataset, _ = random_split(train_dataset, 
            [int(select_rate * len(train_dataset)), len(train_dataset) - int(select_rate * len(train_dataset))])
        val_dataset, _ = random_split(val_dataset, 
            [int(select_rate * len(val_dataset)), len(val_dataset) - int(select_rate * len(val_dataset))])
        
                    
        return train_dataset, val_dataset


    def training_init(self):
        model = None
        if self.cfg.iter_now == 0:
            model = FreckersNet()
        else:
            model = torch.load(
                self.cfg.model_base_dir 
                + "\\" + str(self.cfg.iter_now) + ".pth")
            
        train_dataset, val_dataset = self.load_dataset()

        self.trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            modelPath=self.cfg.model_base_dir\
                + "\\" + str(self.cfg.iter_now + 1) + ".pth")
        
        # 500k <- 25k
    def start(self):
        for i in range(self.cfg.iter_rounds):
            print(f"[IterManager]: Iter {i+1} Start.")
            self.simulation_init()
            print("[IterManager]: iSimulation Init Finish")

            self.simulator.run(self.cfg.simulation_round)

            print("[IterManager]: Simulation Finish")

            self.training_init()
            
            print("[IterManager]: Training Init Finish")

            self.trainer.train(self.cfg.train_config)
            print("[IterManager]: Training Finish")



if __name__ == "__main__":
    im = IterManager()
    im.start()