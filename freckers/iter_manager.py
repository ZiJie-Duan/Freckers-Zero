from global_config import MctsConfig, FreckersConfig


class IterManager:

    def __init__(self) -> None:

        self.mc_cfg = MctsConfig()
        self.fk_cfg = FreckersConfig()

        # 500k <- 25k
    def start(self):
        for i in range(self.fk_cfg.iter_number):
            print(f"[IterManager]: Iter {i+1} Start.")



        






def mcts_data_collect(model, thread_num, file, config, rounds=100, sim_step=300, model2=None):
    deep_frecker = DeepFrecker(model, model2)
    data_record = DataRecord(file=file)

    for j in range(rounds):

        game = Game()
        mcts = MCTS(prob=2, action=(0,0,0,0,False), 
                    game=game, config=config, player=1,
                    deep_frecker=deep_frecker, data_record=data_record)

        for i in range(300):
            print("线程", thread_num, "第", j, "轮游戏 ", "第", i, "步 模拟进行中")
            if i > 30:
                mcts.config.t = 0.2
            elif i > 60:
                mcts.config.t = 0.01
            else:
                mcts.config.t = 1
            # if i > 100:
            #     mcts.config.visulze = True
            # else:
            #     mcts.config.visulze = False
            mcts.run_simu(sim_step)
            end, _ = mcts.move()
            if end:
                break


def mcts_competition(model, thread_num, file, config, rounds=100, sim_step=300, model2=None):
    deep_frecker = DeepFrecker(model, model2)
    data_record = DataRecord(file=file)
    winner_record = []

    for j in range(rounds):

        game = Game()
        mcts = MCTS(prob=2, action=(0,0,0,0,False), 
                    game=game, config=config, player=1,
                    deep_frecker=deep_frecker, data_record=data_record)

        for i in range(300):
            print("线程", thread_num, "第", j, "轮游戏 ", "第", i, "步 模拟进行中")
            if i > 30:
                mcts.config.t = 0.2
            elif i > 60:
                mcts.config.t = 0.01
            else:
                mcts.config.t = 1
            # if i > 100:
            #     mcts.config.visulze = True
            # else:
            #     mcts.config.visulze = False
            mcts.run_simu(sim_step)
            end, winner = mcts.move()
            if end:
                winner_record.append(winner)
                break
    
    return winner_record

