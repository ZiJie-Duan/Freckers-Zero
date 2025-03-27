
class Simulator:

    def __init__(self, game, mcts_agent, dataRecorder) -> None:
        self.game = game
        self.mcts_agent = mcts_agent
        self.dataRecorder = dataRecorder

    def play(self):
        self.game.init()
        while True:
            self.mcts_agent.simulate(self.game.copy())
            pi = self.mcts_agent.getPi()
            action = self.mcts_agent.getAction(pi)
            player = self.mcts_agent.getPlayer()
            s,r,sn,end = self.game.step(player, *action)

            self.dataRecorder.add(s,pi,0)

            if end:
                self.dataRecorder.update_value_and_save(r)
                break

    def run(self, round):
        for i in range(round):
            self.paly()
    