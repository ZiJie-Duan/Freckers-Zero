use pyo3::prelude::*;
use crate::game::{Game, Action, Direction, Player, CellType};

#[pyclass]
struct MctsAcc{
    game: Game,
    dirs: [Direction; 8],
}

#[pymethods]
impl MctsAcc {

    #[new]
    fn new() -> Self{
        MctsAcc { 
            game: Game::new(),
            dirs: [Direction::Up, Direction::Down, Direction::Left, Direction::Right, Direction::UpLeft, Direction::UpRight, Direction::DownLeft, Direction::DownRight],
        }
    }

    
    fn get_action_space(&mut self){
        let gb = self.game.get_game_board();

        for row in gb{
            for cell in row{
                if cell == CellType::BlueFrog || cell == CellType::RedFrog{
                }
            }
        }


    }
}