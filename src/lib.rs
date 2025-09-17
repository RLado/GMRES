//! # GMRES: Generalized minimum residual method
//!
//! A sparse linear system solver using the GMRES iterative method.
//!
//! ![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rlado/GMRES/rust.yml) [![Crates.io](https://img.shields.io/crates/d/gmres)](https://crates.io/crates/gmres) [![Crates.io](https://img.shields.io/crates/v/gmres)](https://crates.io/crates/gmres)
//!
//! ---
//!
//! This crates provides a solver for `Ax=b` linear problems using the GMRES method.
//! Sparse matrices are a common representation for many real-world problems commonly
//! found in engineering and scientific applications. This implementation of the
//! GMRES method is specifically tailored to sparse matrices, making it an efficient
//! and effective tool for solving large linear systems arising from real-world
//! problems.
//!
//! ## Example:
//! ### Solve a linear system
//! ```rust
//! // Define an arbitrary matrix `A`
//! let a = rsparse::data::Sprs::new_from_vec(&[
//!     vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
//!     vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
//!     vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
//!     vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
//!     vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
//! ]);
//!
//! // Define a vector `b`
//! let b = vec![0.104594, 0.437549, 0.040264, 0.298842, 0.254451];
//!
//! // Provide an initial guess
//! let mut x = vec![0.; b.len()];
//!
//! // Solve for `x`
//! gmres::gmres(&a, &b, &mut x, 100, 1e-5).unwrap();
//!
//! // Check if the result is correct
//! gmres::test_utils::assert_eq_f_vec(
//!     &x,
//!     &vec![0.037919, 0.888551, -0.657575, -0.181680, 0.292447],
//!     1e-5,
//! );
//! ```

use rsparse::data::Sprs;
pub mod dense_math;

/// Get a `Sprs` single column from a Sprs matrix
/// # Parameters:
///    - a: `Sprs` matrix
///    - col: Column index
///
fn get_sprs_col(a: &Sprs<f64>, col: usize) -> Sprs<f64> {
    let mut r: Sprs<f64> = Sprs::new();

    // Set parameters
    r.nzmax = (a.p[col + 1] - a.p[col]) as usize;
    r.m = a.m;
    r.n = 1;

    // Set column pointers
    r.p.push(0);
    r.p.push(r.nzmax as isize);

    // Copy column data
    for i in a.p[col]..a.p[col + 1] {
        r.i.push(a.i[i as usize]);
        r.x.push(a.x[i as usize]);
    }

    r
}

/// Add column matrix to dense matrix
///
/// Adds `cm` into the last column of `a`
///
fn add_col_dense(a: &mut [Vec<f64>], cm: &[Vec<f64>]) {
    // Check lengths are the same
    assert_eq!(a.len(), cm.len());

    // Add cm into a
    for i in 0..a.len() {
        a[i].push(cm[i][0]);
    }
}

/// Add a Sprs column matrix to sparse matrix
///
/// Adds `cm` into the last column of `a`
///
fn add_col_sparse(a: &mut Sprs<f64>, cm: &Sprs<f64>) {
    // Check number of rows is the same
    assert_eq!(a.m, cm.m);

    // Add cm into a
    a.p.push(a.p[a.p.len() - 1] + cm.nzmax as isize);
    for i in 0..cm.nzmax {
        a.i.push(cm.i[i]);
        a.x.push(cm.x[i]);
    }
    a.nzmax += cm.nzmax;
    a.n += 1;
}

/// Norm 2 of a `Sprs` column matrix
///
fn norm2(v: &Sprs<f64>) -> f64 {
    let mut r: f64 = 0.0;
    for val in &v.x {
        r += val.powi(2);
    }
    r.sqrt()
}

/// Norm 2 of a [f64]
///
fn norm2_vec(v: &[f64]) -> f64 {
    let mut r: f64 = 0.0;
    for i in v.iter() {
        r += i.powi(2);
    }
    r.sqrt()
}

/// Arnoldi decomposition for sparse matrices
///
fn arnoldi(a: &Sprs<f64>, q: &Sprs<f64>, k: usize) -> (Vec<f64>, Sprs<f64>) {
    let mut qv = a * get_sprs_col(q, k); // Krylov vector
    let mut h = Vec::with_capacity(k + 2);

    for i in 0..=k {
        let qci = get_sprs_col(q, i);
        let ht = rsparse::transpose(&qv) * &qci;
        if !ht.x.is_empty() && ht.i[0] == 0 {
            h.push(ht.x[0]);
        } else {
            h.push(0.);
        }
        qv = qv - (h[i] * qci);
    }

    h.push(norm2(&qv));
    if h[k + 1] != 0.0 {
        qv = qv / h[k + 1];
    }

    (h, qv)
}

/// Calculate the givens rotation matrix
///
fn givens_rotation(v1: f64, v2: f64) -> (f64, f64) {
    if v2 == 0.0 {
        return (1.0, 0.0);
    }
    if v1 == 0.0 {
        return (0.0, 1.0);
    }
    let t = (v1.powi(2) + v2.powi(2)).sqrt();
    let cs = v1 / t;
    let sn = v2 / t;

    (cs, sn)
}

/// Apply givens rotation to H col
///
fn apply_givens_rotation(h: &mut [f64], cs: &mut [f64], sn: &mut [f64], k: usize) {
    for i in 0..k {
        let temp = cs[i] * h[i] + sn[i] * h[i + 1];
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i] = temp;
    }

    // Update the next sin cos values for rotation
    (cs[k], sn[k]) = givens_rotation(h[k], h[k + 1]);

    // Eliminate H(i+1:i)
    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = 0.0;
}

/// GMRES solver for `Sprs` input matrices. Solves Ax = b. Overwrites x with
/// the solution.
///
/// # Parameters:
/// - a: `Sprs` matrix
/// - b: Dense vector
/// - x: Initial guess. When solving differential equations `b` is generally a
/// good guess
/// - max_iter: Maximum number of iterations
/// - threshold: Threshold for convergence
///
/// # Returns:
/// - Error if the method fails to converge
///
pub fn gmres(
    a: &Sprs<f64>,
    b: &[f64],
    x: &mut Vec<f64>,
    max_iter: usize,
    threshold: f64,
) -> Result<(), String> {
    // Calculate initial residual: r = b - A*x
    let ax = rsparse::gaxpy(a, x, &vec![0.0; a.m]);
    let r: Vec<f64> = b
        .iter()
        .zip(ax.iter())
        .map(|(b_i, ax_i)| b_i - ax_i)
        .collect();

    let b_norm = norm2_vec(b);
    if b_norm == 0.0 {
        // b is the zero vector; the solution is x = 0.
        for val in x.iter_mut() {
            *val = 0.0;
        }
        return Ok(());
    }

    let r_norm = norm2_vec(&r);
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors
    let mut sn = vec![0.; max_iter];
    let mut cs = vec![0.; max_iter];
    let mut e1 = vec![0.; max_iter + 1];
    e1[0] = 1.;
    let mut e = vec![error];
    let q_col = dense_math::scxvec(r_norm.powi(-1), &r);
    let mut q: Sprs<f64> = Sprs::new_from_vec(&dense_math::transpose(&[q_col]));
    let mut beta = dense_math::scxvec(r_norm, &e1);
    let mut hs = Vec::with_capacity(max_iter); //Store hessenberg vectors

    let mut ks = 0;
    for k in 0..max_iter {
        ks = k;

        // Arnoldi
        let (mut h, qv) = arnoldi(a, &q, k);
        add_col_sparse(&mut q, &qv);

        // Eliminate the last element in H ith row and update the rotation matrix
        apply_givens_rotation(&mut h, &mut cs, &mut sn, k);
        hs.push(h.clone());

        // Update the residual vector
        beta[k + 1] = -sn[k] * beta[k];
        beta[k] *= cs[k];
        error = f64::abs(beta[k + 1]) / b_norm;

        // Save the error
        e.push(error);

        if error <= threshold {
            break;
        }
    }

    // Form H matrix from the stored columns
    let col_len = ks + 1;
    let mut hm = vec![vec![]; col_len];
    for hi in hs.iter().take(col_len) {
        let mut th = hi.clone();
        th.resize(col_len, 0.0);
        add_col_dense(&mut hm, &dense_math::transpose(&[th[..col_len].to_vec()]));
    }

    // Reduce Q to Q(:, 1:k)
    q.p = q.p[..col_len + 1].to_vec();
    q.n = col_len;
    q.nzmax = (q.p[col_len] - q.p[0]) as usize;

    // Calculate the result by solving the upper triangular system
    let mut y = beta[..col_len].to_vec();
    let hms: Sprs<f64> = rsparse::data::Sprs::new_from_vec(&hm);
    rsparse::usolve(&hms, &mut y);
    *x = rsparse::gaxpy(&q, &y, x);

    if error <= threshold {
        Ok(())
    } else {
        Err(format!(
            "GMRES did not converge. Error: {}. Threshold: {}",
            error, threshold
        ))
    }
}

// --- Unit tests --------------------------------------------------------------
pub mod test_utils;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn norm2_1() {
        let a = Sprs::new_from_vec(&[
            vec![0.888641],
            vec![0.695741],
            vec![0.149974],
            vec![0.429292],
            vec![0.454428],
        ]);
        let n = norm2(&a);
        assert!((n - 1.29885603324079).abs() < 1e-9);
    }

    #[test]
    fn norm2_vec_1() {
        let a = vec![0.888641, 0.695741, 0.149974, 0.429292, 0.454428];
        let n = norm2_vec(&a);
        assert!((n - 1.29885603324079).abs() < 1e-9);
    }

    #[test]
    fn arnoldi_1() {
        let a = Sprs::new_from_vec(&[
            vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
        ]);
        let q = Sprs::new_from_vec(&[
            vec![-0.491347],
            vec![-0.200666],
            vec![-0.817626],
            vec![-0.137704],
            vec![-0.175601],
        ]);

        let (h, qv) = arnoldi(&a, &q, 0);

        test_utils::assert_eq_f_vec(&h, &vec![2.077054, 1.022011], 1e-5);
        test_utils::assert_eq_f2d_vec(
            &qv.to_dense(),
            &vec![
                vec![-0.280376],
                vec![-0.817181],
                vec![0.437209],
                vec![-0.146969],
                vec![-0.202122],
            ],
            1e-5,
        );
    }

    #[test]
    fn arnoldi_2() {
        let a = Sprs::new_from_vec(&[
            vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
            vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
            vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
            vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
            vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
        ]);
        let q = Sprs::new_from_vec(&[
            vec![-0.491347, -0.280376, 0.396178, 0.585492],
            vec![-0.200666, -0.817181, 0.078428, -0.284736],
            vec![-0.817626, 0.437209, -0.041516, -0.246211],
            vec![-0.137704, -0.146969, -0.848475, 0.474661],
            vec![-0.175601, -0.202122, -0.339498, -0.538704],
        ]);

        let (h, qv) = arnoldi(&a, &q, 3);

        test_utils::assert_eq_f_vec(
            &h,
            &vec![0.364447, -0.084894, -0.297025, 0.312162, 0.107295],
            1e-5,
        );
        test_utils::assert_eq_f2d_vec(
            &qv.to_dense(),
            &vec![
                vec![0.424511],
                vec![-0.452464],
                vec![-0.279270],
                vec![-0.119267],
                vec![0.723084],
            ],
            1e-5,
        );
    }
}
