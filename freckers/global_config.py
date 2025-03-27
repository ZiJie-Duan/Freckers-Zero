
class MctsConfig:
    def __init__(self) -> None:
        self.c = 1.5
        self.t = 1
        self.finish = False
        self.visulze = False
        self.small = 0.0000001

        self.dirichlet_alpha = 0.03
        self.dirichlet_epsilon = 0.25 

        self.search_step = 200

class FreckersConfig:
    
    def __init__(self) -> None:
        # iter setting
        self.iter_number = 20

        # simulation setting
        self.simulation_round = 150
        self.simulation_thred = 3

        # mcts setting
        self.mcts_config = MctsConfig()
        self.init_player = 0

        # model / dataset setting
        self.model_dir = ""
        self.dataset_dir = ""

        # game setting
        self.rounds_limit = 250

        




