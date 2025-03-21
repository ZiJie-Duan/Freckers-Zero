use std::cell::Cell;

use pyo3::prelude::*;

#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CellType {
    RedFrog,
    BlueFrog,
    LotusLeaf,
    Empty,
}

impl From<Player> for CellType {
    fn from(player: Player) -> Self {
        match player {
            Player::Red => CellType::RedFrog,
            Player::Blue => CellType::BlueFrog,
        }
    }
}

#[pyclass]
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Player {
    Red,
    Blue,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

impl Direction {
    pub fn goFromLoc(&self, row:i8, col: i8) -> (i8, i8) {
        match self {
            Direction::Up => (row - 1, col),
            Direction::Down => (row + 1, col),
            Direction::Left => (row, col - 1),
            Direction::Right => (row, col + 1),
            Direction::UpLeft => (row - 1, col - 1),
            Direction::UpRight => (row - 1, col + 1),
            Direction::DownLeft => (row + 1, col - 1),
            Direction::DownRight => (row + 1, col + 1),
        }
    }

    fn new(dir: i8) -> Self {
        let direction = match dir {
            0 => Direction::Up,
            1 => Direction::UpRight,
            2 => Direction::Right,
            3 => Direction::DownRight,
            4 => Direction::Down,
            5 => Direction::DownLeft,
            6 => Direction::Left,
            7 => Direction::UpLeft,
            _ => panic!("Error dir Value"),
        };
        direction
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Action {
    Move { row: i8, col: i8, dir: Direction },
    Grow(),
}

impl Action {
    pub fn new(row:i8, col:i8, dir:i8, grow:bool) -> Self{
        if grow{
            Action::Grow()
        }else {
            Action::Move{row:row, col:col, dir: Direction::new(dir)}
        }
    } 
}

#[derive(Debug)]
pub struct Game {
    gameBoard: [[CellType; 8]; 8],
    round: Player,
    round_counter: i64,
}

impl Game { 
    fn grow(&mut self, player:Player){

        let aim_type:CellType = match player {
            Player::Blue => CellType::BlueFrog,
            Player::Red => CellType::RedFrog,
        };

        let dirs:[Direction; 8] = [Direction::Up, Direction::Down, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight, Direction::DownLeft, Direction::DownRight];


        for row in 0..8 {
            for col in 0..8 {
                if self.gameBoard[row][col] == aim_type {

                    for direction in dirs.iter() {
                        let (new_row, new_col) = direction.goFromLoc(row as i8 , col as i8);
                        if new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8 {
                            if self.gameBoard[new_row as usize][new_col as usize] == CellType::Empty{
                                self.gameBoard[new_row as usize][new_col as usize] = CellType::LotusLeaf;
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn check_win(&self) -> Option<Player> {
        let mut c = 0;
        for col in 0..8 {
            if self.gameBoard[0][col] == CellType::BlueFrog {
                c += 1;
            }
        }
        if c == 6 {
            return Some(Player::Blue);
        }
        
        let mut c = 0;
        for col in 0..8 {
            if self.gameBoard[7][col] == CellType::RedFrog {
                c += 1;
            }
        }
        if c == 6 {
            return Some(Player::Red);
        }

        return None;
    }

    pub fn is_valid_move(&self, player:Player, action:Action) -> bool {

        if let Action::Move { row, col, dir } = action {
            if row < 0 || col < 0{
                return false;
            }
            if self.gameBoard[row as usize][col as usize] != player.into(){
                return false;
            }
            match player {
                Player::Blue => {
                    if dir == Direction::Down || dir == Direction::DownLeft || dir == Direction::DownRight{
                        return false;
                    }
                }
                Player::Red => {
                    if dir == Direction::Up || dir == Direction::UpLeft || dir == Direction::UpRight{
                        return false;
                    }
                }
            }
            let (new_row, new_col) = dir.goFromLoc(row, col);
            if !(new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8){
                return false;
            }
            if self.gameBoard[new_row as usize][new_col as usize] != CellType::LotusLeaf{
                if self.gameBoard[new_row as usize][new_col as usize] == CellType::Empty{
                    return false;
                } else {
                    let (new_row, new_col) = dir.goFromLoc(new_row, new_col);
                    if self.gameBoard[new_row as usize][new_col as usize] != CellType::LotusLeaf{
                        return false;
                    }
                }
            }
            return true;
        } else {
            return false;
        }
    }

    fn init_game_board() ->[[CellType; 8]; 8] {
        let mut game_board = [[CellType::Empty; 8]; 8];
        for i in 1..7{
            game_board[0][i] = CellType::RedFrog;
            game_board[7][i] = CellType::BlueFrog;
            game_board[1][i] = CellType::LotusLeaf;
            game_board[6][i] = CellType::LotusLeaf;
        }
        game_board[0][0] = CellType::LotusLeaf;
        game_board[0][7] = CellType::LotusLeaf;
        game_board[7][0] = CellType::LotusLeaf;
        game_board[7][7] = CellType::LotusLeaf;
        game_board
    }

    fn eval(&self, player: Player) -> f64 {
        let mut total_distance = 0;
        let max_distance = 35.0;

        for row in 0..8 {
            for col in 0..8 {
                if self.gameBoard[row][col] == player.into() {
                    let distance = match player {
                        Player::Blue => row,
                        Player::Red => 7 - row,
                    };
                    total_distance += distance;
                }
            }
        }

        let normalized_distance = total_distance as f64 / max_distance;
        1.0 - normalized_distance
    }

}

impl Game {

    pub fn new() -> Self {
        Game {
            gameBoard: Self::init_game_board(),
            round: Player::Red,
            round_counter: 0,
        }
    }


    pub fn step(&mut self, player:Player, action:Action)
    -> ([[CellType; 8]; 8], Action, [[CellType; 8]; 8], f32, bool, bool){

        let mut s: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut sn: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut valid = false;
        let mut r: f32;
        let mut end = false;

        if self.round == player{
            
            if let Action::Move { row, col, dir } = action{

                if self.is_valid_move(player.clone(), action.clone()){
                    let (mut nrow, mut ncol) = dir.goFromLoc(row, col);
                    if self.gameBoard[nrow as usize][ncol as usize] != CellType::LotusLeaf{
                        (nrow, ncol) = dir.goFromLoc(nrow, ncol);
                    }
                    self.gameBoard[nrow as usize][ncol as usize] = player.into();
                    self.gameBoard[row as usize][col as usize] = CellType::Empty;
                    valid = true;
                    sn = self.gameBoard.clone();            
                } 

            } else {
                self.grow(player);
                valid = true;
            } 
        }

        r = match self.check_win(){
            Some(p) => {
                end = true;
                if p == player{
                    1 as f32
                } else {
                    -1 as f32
                }
            }   ,
            None => 0 as f32,
        };
        return (s, action, sn, r, end, valid);
    } 
    
    
    pub fn pprint(&self){
        println!("\nGameBoard:");
        for row in self.gameBoard.iter() {
            for cell in row.iter() {
                let symbol = match cell {
                    CellType::RedFrog => "🔴",
                    CellType::BlueFrog => "🔵",
                    CellType::LotusLeaf => "🟢",
                    CellType::Empty => "⚪",
                };
                print!("{}", symbol);
            }
            println!();
        }
    }

    pub fn init(&mut self){
        self.gameBoard = Game::init_game_board();
        self.round = Player::Red;
    }

    pub fn get_game_board(&self) -> &[[CellType; 8]; 8]{
       &self.gameBoard
    }

    pub fn unsafe_move(&mut self, player:Player, r: i8, c: i8, nr:i8, nc:i8)
    -> ([[CellType; 8]; 8], [[CellType; 8]; 8], f32, bool, bool){
        let mut s: [[CellType; 8]; 8] = self.gameBoard.clone();
        self.gameBoard[nr as usize][nc as usize] = player.into();
        self.gameBoard[r as usize][c as usize] = CellType::Empty;
        let mut sn: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut valid = true;
        let mut r: f32;
        let mut end = false; 
        r = match self.check_win(){
            Some(p) => {
                end = true;
                if p == player{
                    1 as f32
                } else {
                    -1 as f32
                }
            }   ,
            None => 0 as f32,
        };

        if self.round_counter >= 30{
            r = self.eval(player) as f32;
            end = true;
        }

        self.round_counter += 1;

        return (s, sn, r, end, valid);
    }
    pub fn unsafe_grow(&mut self, player:Player)
        -> ([[CellType; 8]; 8], [[CellType; 8]; 8], f32, bool, bool){
        let mut s: [[CellType; 8]; 8] = self.gameBoard.clone();
        self.grow(player);
        let mut sn: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut valid = true;
        let mut r: f32;
        let mut end = false; 
        r = match self.check_win(){
            Some(p) => {
                end = true;
                if p == player{
                    1 as f32
                } else {
                    -1 as f32
                }
            }   ,
            None => 0 as f32,
        };

        if self.round_counter >= 30{
            r = self.eval(player) as f32;
            end = true;
        }

        self.round_counter += 1;
        return (s, sn, r, end, valid);
    }

}


impl Clone for Game {
    fn clone(&self) -> Game {
        Game {
            gameBoard: self.gameBoard.clone(),
            round: self.round.clone(),
            round_counter: self.round_counter,
        }
    }
}