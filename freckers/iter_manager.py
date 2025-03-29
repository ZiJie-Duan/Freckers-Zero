from deep_frecker import DeepFrecker
from data_record import DataRecord
from mcts_agent import MCTSAgent
from game import Game
from simulator import Simulator
from model import FreckersNet, FreckerDataSet
from trainer import Trainer
import torch
from torch.utils.data import DataLoader, random_split
import random
import threading
from multiprocessing import Process


class IterManager:

    def __init__(self, freckers_config) -> None:
        self.cfg = freckers_config
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
                + "\\" + str(self.cfg.iter_now) + ".pth", weights_only=False)
        
        deepfrecker = DeepFrecker(model=model)
        datarecorder = DataRecord(
            file=self.cfg.dataset_base_dir 
            + "\\" + str(self.cfg.iter_now + 1) + ".h5")
        
        mcts_agent = MCTSAgent(
            deepfrecker0=deepfrecker,
            deepfrecker1=deepfrecker,
            mcts_config= self.cfg.mcts_config,
            first_player=self.cfg.init_player
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
                now + 2):

                datafiles.append(
                    self.cfg.dataset_base_dir + "\\" + str(i) + ".h5"
                )
        
        datasets = [FreckerDataSet(x) for x in datafiles]
        dataset = torch.utils.data.ConcatDataset(datasets)
        print(len(dataset))

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
                + "\\" + str(self.cfg.iter_now) + ".pth", weights_only=False)
            
        train_dataset, val_dataset = self.load_dataset()

        self.trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            modelPath=self.cfg.model_base_dir\
                + "\\" + str(self.cfg.iter_now + 1) + ".pth")
    
    def simulation_worker(self):
        self.simulation_init()
        if not self.cfg.skip_first_simu:
            self.simulator.run(self.cfg.simulation_round)
            self.cfg.skip_first_simu = False

        # 500k <- 25k
    def start(self):
        for i in range(self.cfg.iter_rounds):
            print(f"[IterManager]: Iter {i+1} Start.")

            self.simulation_init()
            print("[IterManager]: iSimulation Init Finish")

            if not self.cfg.skip_first_simu:
                self.simulator.run(self.cfg.simulation_round)
            self.cfg.skip_first_simu = False

            print("[IterManager]: Simulation Finish")

            self.training_init()
            
            print("[IterManager]: Training Init Finish")

            self.trainer.train(self.cfg.train_config)
            print("[IterManager]: Training Finish")

            self.cfg.iter_now += 1
    
    def compare_model(self):
        model1 = torch.load(r"C:\Users\lucyc\Desktop\models\p35-45.pth", weights_only=False)
        model2 = torch.load(r"C:\Users\lucyc\Desktop\models\p35-45.pth", weights_only=False)
        
        deepfrecker1 = DeepFrecker(model=model1)
        deepfrecker2 = DeepFrecker(model=model2)

        datarecorder = DataRecord(
            file=self.cfg.dataset_base_dir 
            + "\\" + "Test.h5")
        
        mcts_agent = MCTSAgent(
            deepfrecker0=deepfrecker1,
            deepfrecker1=deepfrecker2,
            mcts_config= self.cfg.mcts_config,
            first_player=self.cfg.init_player
        )

        game = Game(self.cfg.game_rounds_limit)

        self.simulator = Simulator(
            game=game, mcts_agent=mcts_agent, dataRecorder=datarecorder
        )

        self.simulator.run(self.cfg.simulation_round)

