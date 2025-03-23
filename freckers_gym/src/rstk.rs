use std::cell::Cell;
use pyo3::{ffi::{printfunc, PyInterpreterState}, prelude::*};
use crate::game::{self, Action, CellType, Direction, Game, Player};
use rand::seq::SliceRandom;

#[pyclass]
pub struct RSTK{
    gameboard: [[i8;8];8],
    dirs: [Direction; 8],
    blue_dirs: [Direction; 5],
    red_dirs: [Direction; 5],
}

impl RSTK{

    fn player_to_dirs(&self, player: i8) -> [Direction; 5]{
        match player {
            1 => self.red_dirs,
            2 => self.blue_dirs,
            _ => {panic!("error player not exist");}
        }
    }

    fn loc_check(&self, row:i8, col:i8)-> bool{
        if row < 0 || row >=8 || col < 0 || col >= 8 {
            return false;
        }
        return true;
    }

    fn iter_action_space_search(&self, row:i8, col:i8, actions:&mut Vec<(i8,i8)>, player:i8, first:bool){

        let dirs = self.player_to_dirs(player);
        actions.push((row,col));

        for dir in dirs {
            // search space around the frog
            let (r,c) = dir.goFromLoc(row, col);
            if self.loc_check(r,c) {
                // in one step not go out of the gameboard

                // if the first move, can move any direction around the forg
                if first && self.gameboard[r as usize][c as usize] == 3 {
                    if !actions.contains(&(r,c)){
                        actions.push(dir.goFromLoc(row, col));
                    }
                
                // if the not the first move, can only do jump
                } else if self.gameboard[r as usize][c as usize] == 1 ||
                 self.gameboard[r as usize][c as usize] == 2 {
                    // search jump position
                    let (r,c) = dir.goFromLoc(r,c);
                    if self.loc_check(r,c) && !actions.contains(&(r,c)){
                        if self.gameboard[r as usize][c as usize] == 3 {
                            self.iter_action_space_search(r, c, actions, player, false);
                        }
                    }
                }
            }
            
        }
    }



}
#[pymethods]
impl RSTK {
    #[new]
    fn new(gameboard:[[[i8;8];8];3]) -> Self{
        // 0: empty, 1: player_1, 2: player_2, 3: Leaf
        let mut gb = [[0;8];8];
        for r in 0..8{
            for c in 0..8{
                if gameboard[0][r][c] == 1{
                    gb[r][c] = 1;
                } else if gameboard[1][r][c] == 1{
                    gb[r][c] = 2;
                } else if gameboard[2][r][c] == 1{
                    gb[r][c] = 3;
                } else {
                    gb[r][c] = 0;
                }
            }
        }

        RSTK { 
            gameboard: gb,
            dirs: [Direction::Up, Direction::Down, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight, Direction::DownLeft, Direction::DownRight],
            blue_dirs: [Direction::Up, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight],
            red_dirs: [Direction::Down, Direction::Left, Direction::Right, Direction::DownLeft, Direction::DownRight],
        }
    }

    fn get_action_space(&mut self, player: i8) -> Vec<Vec<(i8, i8)>>{
        let mut action_space: Vec<Vec<(i8, i8)>> = Vec::new();
        let player = player + 1; //transfer to rust gameboard code

        for row in 0..8{
            for col in 0..8{
                if self.gameboard[row][col] == player{
                    let mut actions:Vec<(i8,i8)> = Vec::new();
                    self.iter_action_space_search(row as i8, col as i8, &mut actions, player, true);
                    if actions.len() != 1{
                        action_space.push(actions);
                    }
                }
            }
        }
        return action_space;
    }

    pub fn pprint(&self){
        println!("\nRust Build-In GameBoard:");
        for r in 0..8{
            for c in 0..8{
                if self.gameboard[r][c] == 1{
                    print!("{}", "🔴");
                } else if self.gameboard[r][c] == 2{
                    print!("{}", "🔵");
                } else if self.gameboard[r][c] == 3{
                    print!("{}", "🟢");
                } else {
                    print!("{}", "⚪");
                }
            }
            println!("");
        }
    }

}