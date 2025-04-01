from iter_manager import IterManager
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


class MctsConfig:
    def __init__(self) -> None:
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

        self.dirichlet_alpha = 0.1
        self.dirichlet_epsilon = 0.25

        self.pb_c_base = 1000
        self.pb_c_init = 3

        self.search_step = 200


class TrainingConfig:
    def __init__(self):
        self.batch_size = 128
        self.shuffle = True
        self.num_workers = 1
        self.epochs = 1
        self.max_l_rate = 0.0001
        self.min_l_rate = 0.00001

 
class FreckersConfig:
    
    def __init__(self) -> None:
        self.visulze = False
        # iter setting
        self.iter_rounds = 2000
        # after 36, no gravity anymore
        self.iter_now = 0
        self.skip_first_simu = True

        # simulation settingss
        self.simulation_round = 35
        self.simulation_thread = 3

        # mcts setting
        self.mcts_config = MctsConfig()
        self.init_player = 0

        # model / dataset setting
        self.model_base_dir = r"C:\Users\lucyc\Desktop\models"
        self.dataset_base_dir = r"C:\Users\lucyc\Desktop\data"

        # game setting
        self.game_rounds_limit = 250

        # training setting
        self.training_dataset_cross = 20 # +2
        self.training_dataset_select_rate = 0.007
        self.training_dataset_eval_rate = 0.98
        self.train_config = TrainingConfig()



class IterManagerMultiProcess(IterManager):
    def __init__(self, freckers_config: FreckersConfig, thread_number: int) -> None:
        super().__init__(freckers_config)
        self.thread_number = thread_number

    def mp_train(self):
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Training Init")
        self.training_init()
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Training Init Finish")
        train_dataset, val_dataset = self.load_dataset()
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Training Dataset Load Finish")
        self.trainer.train(self.cfg.train_config, 
            self.cfg.model_base_dir + "\\" + str(self.cfg.iter_now + 1) + ".pth",
            train_dataset, val_dataset)
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Training Finish")
        self.trainer = None

    def mp_simu(self):
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Simulation Init")
        self.simulation_init()
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Simulation Init Finish")
        if not self.cfg.skip_first_simu:
            self.simulator.run(self.cfg.simulation_round)
            self.cfg.skip_first_simu = False
        print(f"[IterManagerMultiProcess:{self.thread_number}]: Simulation Finish")

        
    def compare_model(self):
        # model1 = torch.load(r"C:\Users\lucyc\Desktop\np2.pth", weights_only=False)
        # model2 = torch.load(r"C:\Users\lucyc\Desktop\np1.pth", weights_only=False)

        # check1 = torch.load(r"C:\Users\lucyc\Desktop\models\3.pth", weights_only=False)
        # model1 = FreckersNet()
        # model1.load_state_dict(check1['model_state_dict'])
        model1 = FreckersNet()
        check2 = torch.load(r"C:\Users\lucyc\Desktop\models\24.pth", weights_only=False)
        model2 = FreckersNet()
        model2.load_state_dict(check2['model_state_dict'])
        
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
            game=game, mcts_agent=mcts_agent, dataRecorder=datarecorder, visulze=self.cfg.visulze
        )

        self.simulator.run(self.cfg.simulation_round)


import multiprocessing

def run_simulation(thread_number: int, cfg: FreckersConfig):
    im = IterManagerMultiProcess(cfg, thread_number)
    im.mp_simu()

def main():
    import time
    cfg = FreckersConfig()
    im = IterManagerMultiProcess(cfg, 0)

    for i in range(cfg.iter_rounds):
        print(f"[Main]: Iteration {i}")

        if not cfg.skip_first_simu:
            simulation_processes = []
            for j in range(cfg.simulation_thread):
                simulation_process = multiprocessing.Process(target=run_simulation, args=(j, cfg))
                simulation_process.start()
                simulation_processes.append(simulation_process)
                time.sleep(1)
            for process in simulation_processes:
                process.join()
        cfg.skip_first_simu = False

        print(f"[Main]: Training Round {i}")
        im.mp_train()

        print(f"[Main]: Training Round {i} Finish")
        cfg.iter_now += 1

if __name__ == "__main__":
    # main()
    im = IterManagerMultiProcess(FreckersConfig(), 0)
    im.compare_model()



# note 可以尝试 移除生长 在空间中 防止神经网络 不喜欢 生长策略