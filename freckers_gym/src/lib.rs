use std::cell::Cell;
use game::Player;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod game;
mod mcts_acc;
use mcts_acc::MctsAcc;
mod rstk;
use rstk::RSTK;

/// A Python module implemented in Rust.
#[pymodule]
fn freckers_gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MctsAcc>()?;
    m.add_class::<Player>()?;
    m.add_class::<RSTK>()?;
    Ok(())
}
