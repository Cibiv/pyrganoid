use pyo3::prelude::*;
use polars::prelude::*;
use triple_accel::hamming;

use pyo3_polars::{
    PyDataFrame,
};
use pyo3_polars::error::PyPolarsErr;


fn hamm(counter: &mut Vec<u32>, v: &str, comp: &Series) {
    for b in comp.iter() {
        let x = hamming(v.as_bytes(), b.get_str().unwrap().as_bytes());
        counter[x as usize] += 1;
    }
}

#[pyfunction]
fn compute_hamming(pydf1: PyDataFrame, pydf2: PyDataFrame, col1: &str, col2: &str, length: usize) -> PyResult<PyDataFrame> {
    let df1: DataFrame  = pydf1.into();
    let df2: DataFrame  = pydf2.into();

    let mut count = Vec::with_capacity(length);
    count.resize(length, 0);

    for a in df1[col1].iter() {
        let v: &str = a.get_str().unwrap();
        hamm(&mut count, v, &df2[col2]);
    }

    let mut dist = Vec::with_capacity(length);
    for i in 0..length {
        dist.push(i as u32);
    }

    let dist_s: Series = Series::new("distance", dist);
    let count_s: Series = Series::new("count", count);

    let df_r : DataFrame = DataFrame::new(vec![dist_s, count_s]).map_err(PyPolarsErr::from)?;
    
    return Ok(PyDataFrame(df_r));
}

/// A Python module implemented in Rust.
#[pymodule]
fn rham(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_hamming, m)?)?;
    Ok(())
}
