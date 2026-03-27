use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use ndarray::Array1;
use rayon::prelude::*;

/// Batch inverse dynamics: compute joint torques for N timesteps in parallel.
#[pyfunction]
fn inverse_dynamics_batch<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,      // (N, n_joints)
    qd: PyReadonlyArray2<'py, f64>,     // (N, n_joints)
    qdd: PyReadonlyArray2<'py, f64>,    // (N, n_joints)
    masses: PyReadonlyArray1<'py, f64>, // (n_segments,)
    lengths: PyReadonlyArray1<'py, f64>,// (n_segments,)
    gravity: f64,
) -> PyResult<&'py PyArray2<f64>> {
    let n = q.shape()[0];
    let n_joints = q.shape()[1];
    let q = q.as_array();
    let qd = qd.as_array();
    let qdd = qdd.as_array();
    let m = masses.as_slice()?;
    let l = lengths.as_slice()?;

    let mut torques = vec![0.0f64; n * n_joints];

    // Parallel over timesteps using rayon
    torques.par_chunks_mut(n_joints)
        .enumerate()
        .for_each(|(i, row)| {
            // Recursive Newton-Euler for each timestep
            for j in (0..n_joints).rev() {
                let angle: f64 = q[[i, j]];
                let vel: f64 = qd[[i, j]];
                let acc: f64 = qdd[[i, j]];
                let inertia = m[j] * l[j] * l[j] / 3.0;
                row[j] = inertia * acc + m[j] * gravity * l[j] * 0.5 * angle.cos();
                if j + 1 < n_joints {
                    row[j] += row[j + 1]; // Propagate child torques
                }
            }
        });

    Ok(PyArray2::from_vec2(py, &torques.chunks(n_joints).map(|c| c.to_vec()).collect::<Vec<_>>())?)
}

/// Batch center-of-mass computation.
#[pyfunction]
fn com_batch<'py>(
    py: Python<'py>,
    q: PyReadonlyArray2<'py, f64>,
    masses: PyReadonlyArray1<'py, f64>,
    lengths: PyReadonlyArray1<'py, f64>,
) -> PyResult<&'py PyArray2<f64>> {
    let n = q.shape()[0];
    let n_joints = q.shape()[1];
    let q = q.as_array();
    let m = masses.as_slice()?;
    let l = lengths.as_slice()?;
    let total_mass: f64 = m.iter().sum();

    let mut com = vec![0.0f64; n * 2]; // (N, 2) for x,z

    com.par_chunks_mut(2)
        .enumerate()
        .for_each(|(i, row)| {
            let mut x_sum = 0.0;
            let mut z_sum = 0.0;
            let mut cum_x = 0.0;
            let mut cum_z = 0.0;
            for j in 0..n_joints {
                let angle: f64 = q[[i, j]];
                let seg_cx = cum_x + l[j] * 0.5 * angle.sin();
                let seg_cz = cum_z + l[j] * 0.5 * angle.cos();
                x_sum += m[j] * seg_cx;
                z_sum += m[j] * seg_cz;
                cum_x += l[j] * angle.sin();
                cum_z += l[j] * angle.cos();
            }
            row[0] = x_sum / total_mass;
            row[1] = z_sum / total_mass;
        });

    Ok(PyArray2::from_vec2(py, &com.chunks(2).map(|c| c.to_vec()).collect::<Vec<_>>())?)
}

/// Phase interpolation with cubic spline (vectorized).
#[pyfunction]
fn interpolate_phases_rs<'py>(
    py: Python<'py>,
    phase_fractions: PyReadonlyArray1<'py, f64>,
    phase_angles: PyReadonlyArray2<'py, f64>,
    n_frames: usize,
) -> PyResult<&'py PyArray2<f64>> {
    let fracs = phase_fractions.as_slice()?;
    let angles = phase_angles.as_array();
    let n_joints = angles.shape()[1];

    let t_out: Vec<f64> = (0..n_frames).map(|i| i as f64 / (n_frames - 1).max(1) as f64).collect();
    let mut result = vec![0.0f64; n_frames * n_joints];

    result.par_chunks_mut(n_joints)
        .enumerate()
        .for_each(|(i, row)| {
            let t = t_out[i];
            // Find surrounding phases
            let mut idx = 0;
            for k in 0..fracs.len() - 1 {
                if fracs[k] <= t && t <= fracs[k + 1] {
                    idx = k;
                    break;
                }
            }
            let denom = fracs[idx + 1] - fracs[idx];
            let alpha = if denom.abs() < 1e-12 { 0.0 } else { (t - fracs[idx]) / denom };
            for j in 0..n_joints {
                row[j] = angles[[idx, j]] + alpha * (angles[[idx + 1, j]] - angles[[idx, j]]);
            }
        });

    Ok(PyArray2::from_vec2(py, &result.chunks(n_joints).map(|c| c.to_vec()).collect::<Vec<_>>())?)
}

#[pymodule]
fn mujoco_models_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(inverse_dynamics_batch, m)?)?;
    m.add_function(wrap_pyfunction!(com_batch, m)?)?;
    m.add_function(wrap_pyfunction!(interpolate_phases_rs, m)?)?;
    Ok(())
}
