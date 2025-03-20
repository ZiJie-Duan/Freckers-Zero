use std::cell::Cell;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

mod game;
mod mcts_acc;

/// A Python module implemented in Rust.
#[pymodule]
fn freckers_gym(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Game>()?;
    m.add_class::<Player>()?;
    m.add_class::<Direction>()?;
    m.add_class::<Action>()?;
    Ok(())
}
