//! # GMRES: Generalized minimum residual method
//!

use rsparse;
use rsparse::data::Sprs;
pub mod dense_math;

/// Get a `Sprs` single column from a Sprs matrix
/// # Parameters:
///    - a: `Sprs` matrix
///    - col: Column index
///    
fn get_sprs_col(a: &Sprs, col: usize) -> Sprs {
    let mut r = Sprs::new();

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

    return r;
}

/// Get a single column from a dense matrix
/// # Parameters:
///    - a: Dense matrix
///    - col: Column index
///    
fn get_dense_col(a: &Vec<Vec<f64>>, col: usize) -> Vec<Vec<f64>> {
    let mut r = Vec::with_capacity(col);

    for i in 0..a.len() {
        r.push(vec![a[i][col]]);
    }

    return r;
}

/// Arnoldi decomposition for sparse matrices
/// 
fn arnoldi(a: &Sprs, q: &Sprs, k: usize) -> (Vec<f64>, Sprs) {
    let mut qv = a * get_sprs_col(&q, k); // Krylov vector
    let mut h = Vec::with_capacity(k + 2);

    for i in 0..=k {
        let qci = get_sprs_col(&q, i);
        let ht = rsparse::transpose(&qv) * &qci;
        if ht.i[0] == 0 {
            h.push(ht.x[0]);
        } else {
            h.push(0.);
        }
        qv = qv - (h[i] * qci);
    }

    h.push(norm2(&qv)); // problem here
    qv = qv/h[k + 1];

    return (h, qv);
}

/// Norm 2 of a `Sprs` column matrix
///
fn norm2(v: &Sprs) -> f64 {
    let mut r = 0.;
    for i in 0..v.x.len() {
        r += v.x[i].powi(2);
    }

    return r.powf(0.5);
}

/// Arnoldi decomposition for dense matrices
///
fn arnoldi_d(a: &Vec<Vec<f64>>, q: &Vec<Vec<f64>>, k: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut qv = dense_math::mul_mat(&a, &get_dense_col(&q, k)); // Krylov vector
    let mut h = Vec::with_capacity(k + 2);

    for i in 0..=k {
        let qci = get_dense_col(&q, i);
        h.push(dense_math::mul_mat(&dense_math::transpose(&qv), &qci)[0][0]);
        qv = dense_math::add_mat(&qv, &dense_math::scxmat(h[i], &qci), 1., -1.);
    }

    h.push(norm2_d(&qv));
    qv = dense_math::scxmat(h[k + 1].powi(-1), &qv);

    return (h, qv);
}

/// Norm 2 of a Vec<Vec<f64>> matrix
///
fn norm2_d(v: &Vec<Vec<f64>>) -> f64 {
    let mut r = 0.;
    for i in 0..v.len() {
        for j in 0..v[0].len() {
            r += v[i][j].powi(2);
        }
    }

    return r.powf(0.5);
}

/// Norm 2 of a Vec<f64>
///
fn norm2_vec(v: &Vec<f64>) -> f64 {
    let mut r = 0.;
    for i in 0..v.len() {
        r += v[i].powi(2);
    }

    return r.powf(0.5);
}

/// Calculate the givens rotation matrix
///
fn givens_rotation(v1: f64, v2: f64) -> (f64, f64) {
    let t = (v1.powi(2) + v2.powi(2)).powf(0.5);
    let cs = v1 / t;
    let sn = v2 / t;

    return (cs, sn);
}

/// Apply givens rotation to H col
///
fn apply_givens_rotation_d(h: &mut Vec<f64>, cs: &mut Vec<f64>, sn: &mut Vec<f64>, k: usize) {
    for i in 0..k {
        let temp = cs[i] * h[i] + sn[i] * h[i + 1];
        h[i + 1] = -sn[i] * h[i] + cs[i] * h[i + 1];
        h[i] = temp;
    }

    // Update the next sin cos values for rotation
    (cs[k], sn[k]) = givens_rotation(h[k], h[k + 1]);

    // Eliminate H(i+1:i)
    h[k] = cs[k] * h[k] + sn[k] * h[k + 1];
    h[k + 1] = 0.;
}

/// GMRES solver for `Sprs` input matrices. Solves Ax = b. Overwrites x with
/// the solution.
///
/// # Parameters:
/// - a: Dense matrix
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
    a: &Sprs,
    b: &Vec<f64>,
    x: &mut Vec<f64>,
    max_iter: usize,
    threshold: f64,
) -> Result<(), String> {
    // Use x as the initial vector
    let r = rsparse::gaxpy(&(-1. * a), &x, &b);

    let b_norm = norm2_vec(&b);
    let r_norm = norm2_vec(&r);
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors (Optimizable?)
    let mut sn = vec![0.; max_iter];
    let mut cs = vec![0.; max_iter];
    let mut e1 = vec![0.; max_iter + 1];
    e1[0] = 1.;
    let mut e = vec![error];
    let mut q = Sprs::new_from_vec(&dense_math::transpose(&vec![dense_math::scxvec(r_norm.powi(-1), &r)]));
    let mut beta = dense_math::scxvec(r_norm, &e1);
    let mut hs = Vec::with_capacity(max_iter); //Store hessemberg vectors

    let mut ks = 0;
    for k in 0..max_iter {
        ks = k;

        // Arnoldi
        let (mut h, qv) = arnoldi(&a, &q, k);
        //hs.push(h.clone());
        add_col_sparse(&mut q, &qv);

        // Eliminate the last element in H ith row and update the rotation matrix
        apply_givens_rotation_d(&mut h, &mut cs, &mut sn, k);
        hs.push(h.clone());

        // Update the residual vector
        beta[k + 1] = -sn[k] * beta[k];
        beta[k] = cs[k] * beta[k];
        error = f64::abs(beta[k + 1]) / b_norm;

        // Save the error
        e.push(error);

        if error <= threshold {
            break;
        }
    }

    // Form H matrix
    let col_len = ks + 1;
    let mut hm = vec![vec![]; col_len];
    for i in 0..col_len {
        let mut th = hs[i].clone();
        // Pad the column with 0s to complete the column length
        for _ in th.len()..col_len {
            th.push(0.);
        }
        // Transpose h and add to hm
        add_col_dense(
            &mut hm,
            &dense_math::transpose(&vec![th[..col_len].to_vec()]),
        );
    }

    // Reduce Q to Q(:, 1:k)
    q.p = q.p[..col_len].to_vec();

    // Calculate the result
    let mut y = beta[..col_len].to_vec();
    let hms = rsparse::data::Sprs::new_from_vec(&hm);
    rsparse::usolve(&hms, &mut y);
    *x = rsparse::gaxpy(&q, &y, &x);

    if error <= threshold {
        Ok(())
    } else {
        Err(format!(
            "GMRES did not converge. Error: {}. Threshold: {}",
            error, threshold
        ))
    }
}

/// GMRES solver for dense input matrices. Solves Ax = b. Overwrites x with
/// the solution.
///
/// # Parameters:
/// - a: Dense matrix
/// - b: Dense vector
/// - x: Initial guess. When solving differential equations `b` is generally a
/// good guess
/// - max_iter: Maximum number of iterations
/// - threshold: Threshold for convergence
///
/// # Returns:
/// - Error if the method fails to converge
///
pub fn gmres_dense(
    a: &Vec<Vec<f64>>,
    b: &Vec<f64>,
    x: &mut Vec<f64>,
    max_iter: usize,
    threshold: f64,
) -> Result<(), String> {
    let bm = vec![b.clone()];

    // Use x as the initial vector
    let r = dense_math::add_mat(
        &dense_math::transpose(&bm),
        &dense_math::mul_mat(&a, &dense_math::transpose(&vec![x.clone()])),
        1.,
        -1.,
    );

    let b_norm = norm2_d(&bm);
    let r_norm = norm2_d(&r);
    let mut error = r_norm / b_norm;

    // Initialize 1D vectors
    let mut sn = vec![0.; max_iter];
    let mut cs = vec![0.; max_iter];
    let mut e1 = vec![0.; max_iter + 1];
    e1[0] = 1.;
    let mut e = vec![error];
    let mut q = dense_math::scxmat(r_norm.powi(-1), &r);
    let mut beta = dense_math::scxvec(r_norm, &e1);
    let mut hs = Vec::with_capacity(max_iter); //Store hessemberg vectors

    let mut ks = 0;
    for k in 0..max_iter {
        ks = k;

        // Arnoldi
        let (mut h, qv) = arnoldi_d(&a, &q, k);
        //hs.push(h.clone());
        add_col_dense(&mut q, &qv);

        // Eliminate the last element in H ith row and update the rotation matrix
        apply_givens_rotation_d(&mut h, &mut cs, &mut sn, k);
        hs.push(h.clone());

        // Update the residual vector
        beta[k + 1] = -sn[k] * beta[k];
        beta[k] = cs[k] * beta[k];
        error = f64::abs(beta[k + 1]) / b_norm;

        // Save the error
        e.push(error);

        if error <= threshold {
            break;
        }
    }

    // Form H matrix
    let col_len = ks + 1;
    let mut hm = vec![vec![]; col_len];
    for i in 0..col_len {
        let mut th = hs[i].clone();
        // Pad the column with 0s to complete the column length
        for _ in th.len()..col_len {
            th.push(0.);
        }
        // Transpose h and add to hm
        add_col_dense(
            &mut hm,
            &dense_math::transpose(&vec![th[..col_len].to_vec()]),
        );
    }

    // Reduce Q to Q(:, 1:k)
    for i in 0..q.len() {
        q[i] = q[i][..col_len].to_vec();
    }

    // Calculate the result
    let mut y = beta[..col_len].to_vec();
    let mut hms = rsparse::data::Sprs::new();
    hms.from_vec(&hm);
    rsparse::usolve(&hms, &mut y);
    *x = dense_math::add_vec(
        &x,
        &dense_math::transpose(&dense_math::mul_mat(&q, &dense_math::transpose(&vec![y])))[0],
        1.,
        1.,
    );

    if error <= threshold {
        Ok(())
    } else {
        Err(format!(
            "GMRES did not converge. Error: {}. Threshold: {}",
            error, threshold
        ))
    }
}

/// Add column matrix to dense matrix
///
/// Adds `cm` into the last column of `a`
///
fn add_col_dense(a: &mut Vec<Vec<f64>>, cm: &Vec<Vec<f64>>) {
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
fn add_col_sparse(a: &mut Sprs, cm: &Sprs) {
    // Check number of rows is the same
    assert_eq!(a.m, cm.m);

    // Add cm into a
    a.p.push(a.p[a.p.len()-1]+cm.nzmax as isize);
    for i in 0..cm.nzmax {
        a.i.push(cm.i[i]);
        a.x.push(cm.x[i]);
    }
}

/// --- Unit tests ------------------------------------------------------------
mod test_utils;
use test_utils::{assert_eq_f2d_vec, assert_eq_f_vec};

#[test]
fn norm2_1() {
    let a = Sprs::new_from_vec(&vec![
        vec![0.888641],
        vec![0.695741],
        vec![0.149974],
        vec![0.429292],
        vec![0.454428],
    ]);

    let n = norm2(&a);

    assert_eq!(n, 1.29885603324079);
}

#[test]
fn norm2_vec_1() {
    let a = vec![
        0.888641,
        0.695741,
        0.149974,
        0.429292,
        0.454428,
    ];

    let n = norm2_vec(&a);

    assert_eq!(n, 1.29885603324079);
}

#[test]
fn arnoldi_1() {
    let a = Sprs::new_from_vec(&vec![
        vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
        vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
        vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
        vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
        vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
    ]);
    let q = Sprs::new_from_vec(&vec![
        vec![-0.491347],
        vec![-0.200666],
        vec![-0.817626],
        vec![-0.137704],
        vec![-0.175601],
    ]);

    let (h, qv) = arnoldi(&a, &q, 0);

    assert_eq_f_vec(&h, &vec![2.077054, 1.022011], 1e-5);
    assert_eq_f2d_vec(
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
    let a = Sprs::new_from_vec(&vec![
        vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
        vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
        vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
        vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
        vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
    ]);
    let q = Sprs::new_from_vec(&vec![
        vec![-0.491347, -0.280376, 0.396178, 0.585492],
        vec![-0.200666, -0.817181, 0.078428, -0.284736],
        vec![-0.817626, 0.437209, -0.041516, -0.246211],
        vec![-0.137704, -0.146969, -0.848475, 0.474661],
        vec![-0.175601, -0.202122, -0.339498, -0.538704],
    ]);

    let (h, qv) = arnoldi(&a, &q, 3);

    assert_eq_f_vec(
        &h,
        &vec![0.364447, -0.084894, -0.297025, 0.312162, 0.107295],
        1e-5,
    );
    assert_eq_f2d_vec(
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

#[test]
fn norm2_d_1() {
    let a = vec![
        vec![0.888641],
        vec![0.695741],
        vec![0.149974],
        vec![0.429292],
        vec![0.454428],
    ];

    let n = norm2_d(&a);

    assert_eq!(n, 1.29885603324079);
}

#[test]
fn arnoldi_d_1() {
    let a = vec![
        vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
        vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
        vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
        vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
        vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
    ];
    let q = vec![
        vec![-0.491347],
        vec![-0.200666],
        vec![-0.817626],
        vec![-0.137704],
        vec![-0.175601],
    ];

    let (h, qv) = arnoldi_d(&a, &q, 0);

    assert_eq_f_vec(&h, &vec![2.077054, 1.022011], 1e-5);
    assert_eq_f2d_vec(
        &qv,
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
fn arnoldi_d_2() {
    let a = vec![
        vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
        vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
        vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
        vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
        vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
    ];
    let q = vec![
        vec![-0.491347, -0.280376, 0.396178, 0.585492],
        vec![-0.200666, -0.817181, 0.078428, -0.284736],
        vec![-0.817626, 0.437209, -0.041516, -0.246211],
        vec![-0.137704, -0.146969, -0.848475, 0.474661],
        vec![-0.175601, -0.202122, -0.339498, -0.538704],
    ];

    let (h, qv) = arnoldi_d(&a, &q, 3);

    assert_eq_f_vec(
        &h,
        &vec![0.364447, -0.084894, -0.297025, 0.312162, 0.107295],
        1e-5,
    );
    assert_eq_f2d_vec(
        &qv,
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
