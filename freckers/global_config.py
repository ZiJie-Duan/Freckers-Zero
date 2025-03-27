
class MctsConfig:
    def __init__(self) -> None:
        self.c = 1.5
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

        self.dirichlet_alpha = 0.03
        self.dirichlet_epsilon = 0.25 

        self.search_step = 50


class TrainingConfig:
    def __init__(self):
        self.batch_size = 32
        self.shuffle = True
        self.num_workers = 2
        self.epochs = 3
        self.max_l_rate = 0.001
        self.min_l_rate = 0.00001


class FreckersConfig:
    
    def __init__(self) -> None:
        # iter setting
        self.iter_rounds = 20
        self.iter_now = 0

        # simulation setting
        self.simulation_round = 1
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
        self.training_dataset_cross = 3
        self.training_dataset_select_rate = 0.3
        self.training_dataset_eval_rate = 0.8
        self.train_config = TrainingConfig()




        




