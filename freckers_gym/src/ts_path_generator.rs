use crate::game::{Game, Action, Direction, Player, CellType};

struct node{
    parent: Option<Box<node>>,

    cost: i32,
}

struct pathGen{
    openset:Vec<node>,
    closeset:Vec<node>,
}

impl pathGen {
    fn new(gameboard: &[[CellType;8];8]){

    }

}

fn cal_cost(r:i8, c:i8, nr:i8, nc:i8) -> i8{
    (r - nr).abs() + (c - nc).abs()
}

fn get_path_nodes(&self, player: Player, r:i8, c:i8, nr:i8, nc:i8) -> Vec<(Direction, i8)>{
    let dirs = self.player_to_dirs(player);
    let mut res: Vec<(Direction, i8)> = Vec::new();
    for dir in dirs{
        let (tr,tc) = dir.goFromLoc(r, c);
        if self.game.get_game_board()[tr as usize][tc as usize] == CellType::LotusLeaf{
            res.push((dir, MctsAcc::cal_cost(tr, tc, nr, nc)));
        }
    }
    res.sort_by(|a, b| a.1.cmp(&b.1));
}
fn iter_path_search(&self){

}

fn path_search(&self, player: Player, r:i8, c:i8, nr:i8, nc:i8) -> Vec<Direction>{
    let dirs = self.player_to_dirs(player);


}