use std::cell::Cell;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

#[pyclass]
#[derive(Copy, Clone, PartialEq, Debug)]
enum CellType {
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

#[pyclass]
#[derive(Clone, Copy, Debug)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
    UpLeft,
    UpRight,
    DownLeft,
    DownRight,
}

#[pymethods]
impl Direction {
    fn goFromLoc(&self, row:i8, loc: i8) -> (i8, i8) {
        match self {
            Direction::Up => (row - 1, loc),
            Direction::Down => (row + 1, loc),
            Direction::Left => (row, loc - 1),
            Direction::Right => (row, loc + 1),
            Direction::UpLeft => (row - 1, loc - 1),
            Direction::UpRight => (row - 1, loc + 1),
            Direction::DownLeft => (row + 1, loc - 1),
            Direction::DownRight => (row + 1, loc + 1),
        }
    }

    #[new]
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
            _ => panic!("无效的方向值"),
        };
        direction
    }
}

#[pyclass]
#[derive(Clone, Copy, Debug)]
enum Action {
    Move { row: i8, col: i8, dir: Direction },
    Grow(),
}

#[pymethods]
impl Action {
    #[new]
    #[pyo3(signature = (row=0, col=0, dir=0, grow = false))]
    fn new(row:i8, col:i8, dir:i8, grow:bool) -> Self{
        if grow{
            Action::Grow()
        }else {
            Action::Move{row:row, col:col, dir: Direction::new(dir)}
        }
    } 
}

#[pyclass]
#[derive(Debug)]
struct Game {
    gameBoard: [[CellType; 8]; 8],
    round: Player,
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

    fn check_win(&self) -> Option<Player> {
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

    fn is_valid_move(&self, player:Player, action:Action) -> bool {

        if let Action::Move { row, col, dir } = action {
            if row < 0 || col < 0{
                return false;
            }
            if self.gameBoard[row as usize][col as usize] != player.into(){
                return false;
            }
            let (new_row, new_col) = dir.goFromLoc(row, col);
            if !(new_row >= 0 && new_row < 8 && new_col >= 0 && new_col < 8){
                return false;
            }
            if self.gameBoard[new_row as usize][new_col as usize] != CellType::LotusLeaf{
                return false;
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

}

#[pymethods]
impl Game {

    #[new]
    fn new() -> Self {
        Game {
            gameBoard: Self::init_game_board(),
            round: Player::Red,
        }
    }


    fn step(&mut self, player:Player, action:Action)
    -> ([[CellType; 8]; 8], Action, [[CellType; 8]; 8], f32, bool, bool){

        let mut s: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut sn: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut valid = false;
        let mut r: f32;
        let mut end = false;

        if let Action::Move { row, col, dir } = action{

            if self.is_valid_move(player.clone(), action.clone()){
                let (nrow, ncol) = dir.goFromLoc(row, col);
                self.gameBoard[nrow as usize][ncol as usize] = player.into();
                self.gameBoard[row as usize][col as usize] = CellType::Empty;
                valid = true;
                sn = self.gameBoard.clone();            
            } 

        } else {
            self.grow(player);
            valid = true;
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
    
    fn help(&self){
        println!("欢迎使用 freckers 游戏的强化学习 gym！\n");
        println!("本程序包含以下函数：\n");
        println!("1. step 函数：用于模拟游戏。\n");
        println!("   参数说明：\n");
        println!("   - player: 玩家，值为 1 表示红色青蛙，值为 2 表示蓝色青蛙。\n");
        println!("   - row: 行坐标，二维数组中的整数。\n");
        println!("   - col: 列坐标，二维数组中的整数。\n");
        println!("   - dir: 青蛙跳动的方向，整数表示。\n");
        println!("   - grow: 布尔值，表示是否跳过动作并成长荷叶。\n");
    }
    
    fn pprint(&self){
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
}

/// A Python module implemented in Rust.
#[pymodule]
fn freckers_gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<Player>()?;
    m.add_class::<Direction>()?;
    m.add_class::<Action>()?;
    Ok(())
}
