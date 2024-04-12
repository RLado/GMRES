# GMRES: Generalized minimum residual method

A sparse linear system solver using the GMRES iterative method.

![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rlado/GMRES/rust.yml) [![Crates.io](https://img.shields.io/crates/d/gmres)](https://crates.io/crates/gmres) [![Crates.io](https://img.shields.io/crates/v/gmres)](https://crates.io/crates/gmres)

---

This crates provides a solver for `Ax=b` linear problems using the GMRES method.
Sparse matrices are a common representation for many real-world problems commonly
found in engineering and scientific applications. This implementation of the 
GMRES method is specifically tailored to sparse matrices, making it an efficient 
and effective tool for solving large linear systems arising from real-world 
problems.

## Example:
### Solve a linear system
```rust
use gmres;
use rsparse::data::Sprs;

fn main() {
    // Define an arbitrary matrix `A`
    let a = Sprs::new_from_vec(&[
        vec![0.888641, 0.477151, 0.764081, 0.244348, 0.662542],
        vec![0.695741, 0.991383, 0.800932, 0.089616, 0.250400],
        vec![0.149974, 0.584978, 0.937576, 0.870798, 0.990016],
        vec![0.429292, 0.459984, 0.056629, 0.567589, 0.048561],
        vec![0.454428, 0.253192, 0.173598, 0.321640, 0.632031],
    ]);

    // Define a vector `b`
    let b = vec![0.104594, 0.437549, 0.040264, 0.298842, 0.254451];

    // Provide an initial guess
    let mut x = vec![0.; b.len()];

    // Solve for `x`
    gmres::gmres(&a, &b, &mut x, 100, 1e-5).unwrap();

    // Check if the result is correct
    gmres::test_utils::assert_eq_f_vec(
        &x,
        &vec![0.037919, 0.888551, -0.657575, -0.181680, 0.292447],
        1e-5,
    );
}
```