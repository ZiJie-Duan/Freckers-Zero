from iter_manager import IterManager

class MctsConfig:
    def __init__(self) -> None:
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

        self.dirichlet_alpha = 0.1
        self.dirichlet_epsilon = 0.25

        self.pb_c_base = 2000
        self.pb_c_init = 3

        self.search_step = 300


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
        # iter setting
        self.iter_rounds = 100
        self.iter_now = 45
        self.skip_first_simu = True

        # simulation setting
        self.simulation_round = 20
        self.simulation_thred = 3

        # mcts setting
        self.mcts_config = MctsConfig()
        self.init_player = 0

        # model / dataset setting
        self.model_base_dir = r"C:\Users\lucyc\Desktop\models"
        self.dataset_base_dir = r"C:\Users\lucyc\Desktop\data"

        # game setting
        self.game_rounds_limit = 250

        # training setting
        self.training_dataset_cross = 18 # +2
        self.training_dataset_select_rate = 0.05
        self.training_dataset_eval_rate = 0.8
        self.train_config = TrainingConfig()



if __name__ == "__main__":
    im = IterManager(FreckersConfig())
    #im.start()
    im.compare_model()




