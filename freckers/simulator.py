import copy
class Simulator:

    def __init__(self, game, mcts_agent, dataRecorder) -> None:
        self.game = game
        self.mcts_agent = mcts_agent
        self.dataRecorder = dataRecorder

    def play(self):
        self.game.init()
        i = 0
        while True:
            if i > 30:
                self.mcts_agent.config.t = 0.2
            elif i > 60:
                self.mcts_agent.config.t = 0.01
            else:
                self.mcts_agent.config.t = 1

            i += 1
            self.mcts_agent.simulate(copy.deepcopy(self.game))
            pi = self.mcts_agent.getPi()
            action = self.mcts_agent.getAction(pi)
            player = self.mcts_agent.getPlayer()

            # pi = [(r,c,rn,cn,v)...] # grow always in the end of the pi
            # action = [(r,c,rn,cn,grow)...] 

            self.game.pprint()
            print(f"player: {player}, action: {action}")

            s,r,sn,end = self.game.step(player, *(action))
            self.mcts_agent.cutMove(pi) 
            self.dataRecorder.add(s,pi,0)

            if end:
                print("end")
                self.dataRecorder.update_value_and_save(r)
                break

    def run(self, round):
        for i in range(round):
            self.mcts_agent.reset()
            self.play()
    