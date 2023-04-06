//! Testing utilities
//! 
//! Borrowed from `rsparse`: <https://github.com/RLado/rsparse>
//! 

/// Assert if A is equal to B within an acceptable margin of error (tol)
/// [Borrowed from `rsparse`: <https://github.com/RLado/rsparse>]
///
pub fn assert_eq_f_vec(a: &Vec<f64>, b: &Vec<f64>, tol: f64) {
    for i in 0..a.len() {
        let diff = f64::abs(a[i] - b[i]);
        if diff > tol {
            panic!(
                "The Vec are not equal: {:?} != {:?}. -- Check failed by: {}",
                a, b, diff
            );
        }
    }
}

#[test]
fn assert_eq_f_vec_1() {
    let a = vec![
        0.856803, -0.024615, 0.629721, -0.123138, 0.195778, -0.450195, -0.628933, 0.636038,
        0.289215, 0.430638,
    ];

    let b = vec![
        0.8568031, -0.024615, 0.629721, -0.123138, 0.195778, -0.450195, -0.628933, 0.636038,
        0.289215, 0.430638,
    ];

    assert_eq_f_vec(&a, &b, 1e-6);
}

#[test]
#[should_panic]
fn assert_eq_f_vec_2() {
    let a = vec![
        0.856803, -0.024615, 0.629721, -0.123138, 0.195778, -0.450195, -0.628933, 0.636038,
        0.289215, 0.430638,
    ];

    let b = vec![
        0.8568031, -0.024615, 0.629721, -0.123138, 0.195778, -0.450195, -0.628933, 0.636038,
        0.289215, 0.430638,
    ];

    assert_eq_f_vec(&a, &b, 1e-7);
}

/// Assert if A is equal to B within an acceptable margin of error (tol)
/// [Borrowed from `rsparse`: <https://github.com/RLado/rsparse>]
///
pub fn assert_eq_f2d_vec(a: &Vec<Vec<f64>>, b: &Vec<Vec<f64>>, tol: f64) {
    for i in 0..a.len() {
        for j in 0..a[0].len() {
            let diff = f64::abs(a[i][j] - b[i][j]);
            if diff > tol {
                panic!(
                    "The 2D Vec are not equal: {:?} != {:?}. -- Check failed by: {}",
                    a, b, diff
                );
            }
        }
    }
}

#[test]
fn assert_eq_f2d_vec_1() {
    let a = vec![
        vec![
            2.9118e-01, 5.6680e-01, 1.8228e-03, 4.0549e-01, 3.8642e-01, 2.5993e-01, 7.8883e-01,
        ],
        vec![
            2.0412e-02, 3.2074e-01, 6.4605e-01, 6.3720e-01, 4.3517e-01, 8.0411e-01, 8.2100e-01,
        ],
        vec![
            4.6343e-01, 8.8938e-01, 6.8361e-01, 2.4497e-01, 2.5148e-01, 9.3315e-01, 8.6388e-01,
        ],
        vec![
            2.2273e-02, 6.2230e-01, 3.5388e-01, 8.8429e-01, 1.4841e-01, 3.5973e-01, 5.5950e-01,
        ],
        vec![
            4.9581e-01, 5.4801e-01, 5.8516e-01, 5.9622e-01, 7.0883e-01, 1.8378e-01, 9.5005e-01,
        ],
        vec![
            2.1346e-01, 1.5274e-01, 6.3519e-02, 2.3448e-01, 1.5056e-01, 6.9372e-01, 6.4248e-02,
        ],
        vec![
            3.1925e-01, 3.7280e-01, 3.7565e-02, 4.6288e-02, 4.8428e-01, 5.1961e-01, 1.8035e-01,
        ],
    ];
    let b = vec![
        vec![
            2.9118e-01, 5.6680e-01, 1.8228e-03, 4.0549e-01, 3.8642e-01, 2.5993e-01, 7.8883e-01,
        ],
        vec![
            2.0412e-02, 3.2074e-01, 6.4605e-01, 6.3720e-01, 4.3517e-01, 8.0411e-01, 8.2100e-01,
        ],
        vec![
            4.6343e-01, 8.8938e-01, 6.8361e-01, 2.4497e-01, 2.5148e-01, 9.3315e-01, 8.6388e-01,
        ],
        vec![
            2.2273e-02, 6.2230e-01, 3.5388e-01, 8.8429e-01, 1.4841e-01, 3.5973e-01, 5.5950e-01,
        ],
        vec![
            4.9581e-01, 5.4801e-01, 5.8516e-01, 5.9622e-01, 7.0883e-01, 1.8378e-01, 9.5005e-01,
        ],
        vec![
            2.1346e-01, 1.5274e-01, 6.3519e-02, 2.3448e-01, 1.5056e-01, 6.9372e-01, 6.4248e-02,
        ],
        vec![
            3.1925e-01,
            3.7280e-01,
            3.7565e-02,
            4.6288e-02,
            4.8428e-01,
            5.19611e-01,
            1.8035e-01,
        ],
    ];

    assert_eq_f2d_vec(&a, &b, 1e-4);
}

#[test]
#[should_panic]
fn assert_eq_f2d_vec_2() {
    let a = vec![
        vec![
            2.9118e-01, 5.6680e-01, 1.8228e-03, 4.0549e-01, 3.8642e-01, 2.5993e-01, 7.8883e-01,
        ],
        vec![
            2.0412e-02, 3.2074e-01, 6.4605e-01, 6.3720e-01, 4.3517e-01, 8.0411e-01, 8.2100e-01,
        ],
        vec![
            4.6343e-01, 8.8938e-01, 6.8361e-01, 2.4497e-01, 2.5148e-01, 9.3315e-01, 8.6388e-01,
        ],
        vec![
            2.2273e-02, 6.2230e-01, 3.5388e-01, 8.8429e-01, 1.4841e-01, 3.5973e-01, 5.5950e-01,
        ],
        vec![
            4.9581e-01, 5.4801e-01, 5.8516e-01, 5.9622e-01, 7.0883e-01, 1.8378e-01, 9.5005e-01,
        ],
        vec![
            2.1346e-01, 1.5274e-01, 6.3519e-02, 2.3448e-01, 1.5056e-01, 6.9372e-01, 6.4248e-02,
        ],
        vec![
            3.1925e-01, 3.7280e-01, 3.7565e-02, 4.6288e-02, 4.8428e-01, 5.1961e-01, 1.8035e-01,
        ],
    ];
    let b = vec![
        vec![
            2.9118e-01, 5.6680e-01, 1.8228e-03, 4.0549e-01, 3.8642e-01, 2.5993e-01, 7.8883e-01,
        ],
        vec![
            2.0412e-02, 3.2074e-01, 6.4605e-01, 6.3720e-01, 4.3517e-01, 8.0411e-01, 8.2100e-01,
        ],
        vec![
            4.6343e-01, 8.8938e-01, 6.8361e-01, 2.4497e-01, 2.5148e-01, 9.3315e-01, 8.6388e-01,
        ],
        vec![
            2.2273e-02, 6.2230e-01, 3.5388e-01, 8.8429e-01, 1.4841e-01, 3.5973e-01, 5.5950e-01,
        ],
        vec![
            4.9581e-01, 5.4801e-01, 5.8516e-01, 5.9622e-01, 7.0883e-01, 1.8378e-01, 9.5005e-01,
        ],
        vec![
            2.1346e-01, 1.5274e-01, 6.3519e-02, 2.3448e-01, 1.5056e-01, 6.9372e-01, 6.4248e-02,
        ],
        vec![
            3.1925e-01,
            3.7280e-01,
            3.7565e-02,
            4.6288e-02,
            4.8428e-01,
            5.19611e-01,
            1.8035e-01,
        ],
    ];
    assert_eq_f2d_vec(&a, &b, 1e-6);
}
