

class MCTS:
    def __init__(self, p, game=None, player=0) -> None:
        self.game = game
        self.player = player
        self.n = 0 # add 1 when backp
        self.w = 0 # add v when backp
        self.q = 0 # 1 / self.n * backp v_acc 
        self.p = p # from parent nn

        self.children = [] #(child, (r,c,nr,nc)/(true,0,0,0)) 
        self.c = 2
        self.t = 1

        self.meta_v = 0

        self.finished = False

    def run(self, step = 100):
        for i in range(step):
            if i > (step - 0):
                self.simu(self.game.dclone(), self.player, visual = True)
            else:
                self.simu(self.game.dclone(), self.player, visual = False)

    def rotate_180(self, row, col):
        new_row = 7 - row
        new_col = 7 - col
        return (new_row, new_col)

    def go(self):
        children_pubt = []
        max = -999
        max_i = 0
        tt = sum([c[0].n**(1/self.t) for c in self.children])
        for i, child_With_action in enumerate(self.children):
            c = child_With_action[0]
            v = (c.n)**(1/self.t) / tt
            children_pubt.append(v)
            if max < v:
                max = v
                max_i = i

        st, fogs_info = self.game.get_game_tensor(self.player)
        if self.player == Player.Blue:
            st = np.rot90(st, 2)
            fogs_info = [(self.rotate_180(loc[0], loc[1]), surface) for loc, surface in fogs_info]

        surface_map = {}
        for loc, surface in fogs_info:
            surface_map[loc] = surface
        pi = [] # (surface, row, col, v) / (True, v)
        for i, data in enumerate(self.children):
            c, a = data
            if a[0] == True:
                pi.append((True, children_pubt[i]))
            else:
                if self.player == Player.Blue:
                    pi.append((surface_map[self.rotate_180(a[0], a[1])], a[2], a[3], children_pubt[i]))
                else:
                    pi.append((surface_map[(a[0], a[1])], a[2], a[3], children_pubt[i]))
        
        memory.append((st, pi, self.children[max_i][0].w / self.children[max_i][0].n))

        r,c,rn,cn = self.children[max_i][1]
        if r == True:
            s,sn,r,end,valid = self.game.step(self.player,0,0,0,0,True)
        else: 
            s,sn,r,end,valid = self.game.step(self.player,r,c,rn,cn,False)
        
        self.game.pprint()
        child = self.children[max_i][0]
        self.player = Player.Red if self.player == Player.Blue else Player.Blue
        self.n = child.n # add 1 when backp
        self.w = child.w # add v when backp
        self.q = child.q # 1 / self.n * backp v_acc 
        self.p = child.p # from parent nn

        self.children = child.children #(child, (r,c,nr,nc)/(true,0,0,0)) 
        self.c = child.c
        self.t = child.t



    def simu(self, game, player, visual = False):
        actions = {}
        surface_map = {}

        if self.n == 0:
            for space in game.get_action_space(player):
                actions[space[0]] = space[1:]

            game_py_tensor, fogs_info = game.get_game_tensor(player)
            #pprint.pprint(game_py_tensor)
            for loc, surface in fogs_info:
                surface_map[loc] = surface

            if player == Player.Red:
                game_tensor = torch.tensor(game_py_tensor)
                result, v = inference(model, game_tensor)
            else:
                game_tensor = torch.tensor(game_py_tensor).flip(dims=[1, 2])
                result, v = inference(model, game_tensor)
                result = np.rot90(result, 2)
            
            v = v[0][0]
            result = result[0]

            for loc, action_group in actions.items():
                for a in action_group:
                    self.children.append(
                        (MCTSFreckerZero(result[surface_map[loc]][a[0]][a[1]]),
                        (loc[0], loc[1], a[0], a[1]))
                    )

            if TC == 1:
                print("\n\n\n Play:", player,  result[6][5][5])
                pprint(result)
                input()

            self.children.append(
                (MCTSFreckerZero(result[6][5][5]),(True,0,0,0))
            )
            max = -999
            min = 999
            for c, a in self.children:
                if c.p > max:
                    max = c.p
                if c.p < min:
                    min = c.p

            for c, a in self.children:
                c.p = (c.p - min)/(max - min + 0.000001)

            if visual == True:
                game.pprint()
            self.n += 1
            self.q = v
            self.meta_v = v
            return v, v
        

        else:

            u_total = 0
            for c in self.children:
                u_total += c[0].n

            max = -999
            max_i = 0
            for i, child_With_action in enumerate(self.children):
                c = child_With_action[0]
                v = self.q + (c.c * c.p * (math.sqrt(u_total)) / (1+c.n))
                if max < v:
                    max = v
                    max_i = i
                
                if TC == 1:
                    print(f"PUBT: {v} CAL PUBT: q:{c.q}, c.p:{c.p} c.n:{c.n} action: {child_With_action[1]}")
            if TC == 1:
                print("Previous:", player)
                input()
            
            #print(f"Choose: {self.children[max_i]}")
            # backp
            r,c,rn,cn = self.children[max_i][1]
            if r == True:
                s,sn,r,end,valid = game.step(player,0,0,0,0,True)
            else: 
                s,sn,r,end,valid = game.step(player,r,c,rn,cn,False)

            if end:
                self.finished = True
                v = r
                vacc = r
            else:
                v, vacc = self.children[max_i][0].simu(game, Player.Red if player == Player.Blue else Player.Blue, visual)

            self.w += v
            self.n += 1
            self.q = vacc + self.meta_v / self.n
            return v, vacc + self.meta_v
        

TC = 0
for _ in range(10):
    memory = []

    for _ in range(1):
        game = MctsAcc()
        mcts = MCTSFreckerZero(1,game)

        for i in range(150):
            print(f"Step {i+1}")
            mcts.run(800)
            if mcts.finished:
                break
            mcts.go()

    train(model, memory, 20)
    model.eval()

    TC += 1

    