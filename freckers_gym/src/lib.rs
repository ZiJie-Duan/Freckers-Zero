use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Player {
    Red,
    Blue,
}

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

impl Direction {
    fn goFromLoc(&self, row: i8, loc: i8) -> (i8, i8) {
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
}

#[derive(Clone, Copy, Debug)]
struct Action {
    row: i8,
    col: i8,
    dir: Direction,
    grow: bool,
}

#[derive(Debug)]
struct Game {
    gameBoard: [[CellType; 8]; 8],
    round: Player,
}

impl Game {
    fn new() -> Self {
        Game {
            gameBoard: [[CellType::Empty; 8]; 8],
            round: Player::Red,
        }
    }

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
                            self.gameBoard[new_row as usize][new_col as usize] = CellType::LotusLeaf;
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
        let row = action.row;
        let col = action.col;
        let dir = action.dir;

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
        if self.gameBoard[new_row as usize][new_col as usize] != CellType::Empty{
            return false;
        }
        true
    }

    fn step(&mut self, player:Player, action:Action)
    -> ([[CellType; 8]; 8], Action, [[CellType; 8]; 8], f32, bool, bool){
        let row = action.row;
        let col = action.col;
        let dir = action.dir;
        let grow = action.grow;

        let mut s: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut sn: [[CellType; 8]; 8] = self.gameBoard.clone();
        let mut valid = false;
        let mut r: f32;
        let mut end = false;

        if grow{
            self.grow(player);
            valid = true;
        } else {
            if self.is_valid_move(player.clone(), action.clone()){
                let (nrow, ncol) = dir.goFromLoc(row, col);
                self.gameBoard[nrow as usize][ncol as usize] = player.into();
                valid = true;
                sn = self.gameBoard.clone();            
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
}


#[pyclass]
struct Freckers{
    game:Game
}

#[pymethods]
impl Freckers {

    #[new]
    fn new() -> Self {
        Freckers{
            game: Game::new(),
        }
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

    fn step(&mut self, player:i8, row:i8, col:i8, dir:i8, grow:bool) -> PyResult<()>{
        let start_time = std::time::Instant::now();

        let player = match player {
            1 => Player::Red,
            2 => Player::Blue,
            _ => {return Err(PyValueError::new_err("player not in range"));}
        };

        let row = if row >= 0 && row < 8 {
            row
        } else {
            return Err(PyValueError::new_err("row not in range"));
        };

        let col = if col >= 0 && col < 8 {
            col
        } else {
            return Err(PyValueError::new_err("col not in range"));
        };

        let dir = match dir {
            0 => Direction::Up,
            1 => Direction::Down,
            2 => Direction::Left,
            3 => Direction::Right,
            4 => Direction::UpLeft,
            5 => Direction::UpRight,
            6 => Direction::DownLeft,
            7 => Direction::DownRight,
            _ => {return Err(PyValueError::new_err("dir not in range"));}
        };
        let a = Action{
            row: row,
            col: col,
            dir: dir,
            grow: grow,
        };
        let (s, a, sn, r, end, v) = self.game.step(player, a);

        let duration = start_time.elapsed();
        println!("代码执行时间: {:?}", duration);
        return Ok(());
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn freckers_gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Freckers>()?;
    Ok(())
}
