use std::cell::Cell;

use pyo3::{ffi::printfunc, prelude::*};
use crate::game::{Game, Action, Direction, Player, CellType};

#[pyclass]
pub struct MctsAcc{
    game: Game,
    dirs: [Direction; 8],
    blue_dirs: [Direction; 5],
    red_dirs: [Direction; 5],
}

impl MctsAcc {

    fn player_to_dirs(&self, player: Player) -> [Direction; 5]{
        match player {
            Player::Blue => self.blue_dirs,
            Player::Red => self.red_dirs,
        }
    }

    fn loc_check(&self, row:i8, col:i8)-> bool{
        if row < 0 || row >=8 || col < 0 || col >= 8 {
            return false;
        }
        return true;
    }

    fn iter_action_space_search(&self, row:i8, col:i8, actions:&mut Vec<(i8,i8)>, player:Player){

        let dirs = self.player_to_dirs(player);

        actions.push((row,col));

        for dir in dirs {
            // search space around the frog
            let (r,c) = dir.goFromLoc(row, col);
            if self.loc_check(r,c) && 
               self.game.get_game_board()[r as usize][c as usize] == CellType::LotusLeaf {
                if !actions.contains(&(r,c)){
                    actions.push(dir.goFromLoc(row, col));
                }
            } else {
                // search jump position
                let (r,c) = dir.goFromLoc(r,c);
                if self.loc_check(r,c) && !actions.contains(&(r,c)){
                    if self.game.get_game_board()[r as usize][c as usize] == CellType::LotusLeaf {
                        self.iter_action_space_search(r, c, actions, player);
                    }
                }
            }
        }
    }

    fn build_tensor(gameboard: [[CellType;8];8], player: Player) -> [[[i32;8];8];8]{
        let tensor = [[[0;8];8];8];



    }


}
#[pymethods]
impl MctsAcc {
    #[new]
    fn new() -> Self{
        MctsAcc { 
            game: Game::new(),
            dirs: [Direction::Up, Direction::Down, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight, Direction::DownLeft, Direction::DownRight],
            blue_dirs: [Direction::Up, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight],
            red_dirs: [Direction::Down, Direction::Left, Direction::Right, Direction::DownLeft, Direction::DownRight],
        }
    }

    fn get_action_space(&mut self, player: Player) -> Vec<Vec<(i8, i8)>>{
        let gb = self.game.get_game_board();
        let mut action_space: Vec<Vec<(i8, i8)>> = Vec::new();

        for row in 0..8{
            for col in 0..8{
                if gb[row][col] == player.into(){
                    let mut actions:Vec<(i8,i8)> = Vec::new();
                    self.iter_action_space_search(row as i8, col as i8, &mut actions, player);
                    if actions.len() != 1{
                        action_space.push(actions);
                    }
                }
            }
        }
        return action_space;
    }

    fn step(&mut self, player: Player, r:i8, c:i8, nr:i8, nc:i8, grow:bool)
    -> ([[CellType; 8]; 8], [[CellType; 8]; 8], f32, bool, bool){
        
        let (s, sn, r, end, valid) 
        = match grow {
            true => self.game.unsafe_grow(player),
            false => self.game.unsafe_move(player, r, c, nr, nc)
        };

        


    }
}