//! Basic dense matrix and vector operations
//!

/// Returns the additive identity element of `Self`, `0`.
pub trait Zero: Sized + std::ops::Add<Self, Output = Self> {
    /// Returns the additive identity element of `Self`, `0`.
    ///
    /// # Laws
    ///
    /// ```{.text}
    /// a + 0 = a       ∀ a ∈ Self
    /// 0 + a = a       ∀ a ∈ Self
    /// ```
    ///
    fn zero() -> Self;

    /// Returns `true` if `self` is equal to the additive identity.
    fn is_zero(&self) -> bool;
}

macro_rules! zero_impl {
    ($t:ty, $v:expr) => {
        impl Zero for $t {
            #[inline]
            fn zero() -> $t {
                $v
            }
            #[inline]
            fn is_zero(&self) -> bool {
                *self == $v
            }
        }
    };
}

zero_impl!(usize, 0usize);
zero_impl!(u32, 0u32);
zero_impl!(u64, 0u64);
zero_impl!(isize, 0isize);
zero_impl!(i32, 0i32);
zero_impl!(i64, 0i64);
zero_impl!(f32, 0.0f32);
zero_impl!(f64, 0.0f64);

/// Trait defining numbers that can perform mathematical operations
///
pub trait Num:
    Zero
    + std::ops::Add
    + std::ops::AddAssign
    + std::ops::Sub
    + std::ops::SubAssign
    + std::ops::Mul
    + std::ops::MulAssign
    + Copy
    + std::fmt::Debug
{
}
impl Num for usize {}
impl Num for u32 {}
impl Num for u64 {}
impl Num for isize {}
impl Num for i32 {}
impl Num for i64 {}
impl Num for f32 {}
impl Num for f64 {}

/// Scalar plus dense matrix
///
/// # Example
/// ```
/// use gmres::dense_math::scpmat;
///
/// let a = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
///
/// let s = 5;
///
/// let c = scpmat(s, &a);
///
/// assert_eq!(&c, &vec![
///     vec![6,7,8],
///     vec![9,10,11],
///     vec![12,13,14],
///     ]
/// );
/// ```
///
pub fn scpmat<T: Num>(scalar: T, mat: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let rows = mat.len();
    let columns = mat[0].len();

    let mut r = vec![vec![T::zero(); columns]; rows];
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            r[i][j] = mat[i][j] + scalar;
        }
    }

    return r;
}

/// Scalar times dense matrix
///
/// # Example
/// ```
/// use gmres::dense_math::scxmat;
///
/// let a = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
///
/// let s = 5;
///
/// let c = scxmat(s, &a);
///
/// assert_eq!(&c, &vec![
///     vec![5, 10, 15],
///     vec![20, 25, 30],
///     vec![35, 40, 45],
///     ]
/// );
/// ```
///
pub fn scxmat<T: Num + std::ops::Mul<Output = T>>(scalar: T, mat: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let rows = mat.len();
    let columns = mat[0].len();

    let mut r = vec![vec![T::zero(); columns]; rows];
    for i in 0..mat.len() {
        for j in 0..mat[0].len() {
            r[i][j] = mat[i][j] * scalar;
        }
    }

    return r;
}

/// Scalar plus dense vector
///
/// # Example
/// ```
/// use gmres::dense_math::scpvec;
///
/// let a = vec![1, 2, 3];
///
/// let s = 5;
///
/// let c = scpvec(s, &a);
///
/// assert_eq!(&c, &vec![6, 7, 8]);
/// ```
///
pub fn scpvec<T: Num>(scalar: T, v: &Vec<T>) -> Vec<T> {
    let mut r = vec![T::zero(); v.len()];
    for i in 0..v.len() {
        r[i] = v[i] + scalar;
    }

    return r;
}

/// Scalar times dense vector
///
/// # Example
/// ```
/// use gmres::dense_math::scxvec;
///
/// let a = vec![1, 2, 3];
///
/// let s = 5;
///
/// let c = scxvec(s, &a);
///
/// assert_eq!(&c, &vec![5, 10, 15]);
/// ```
///
pub fn scxvec<T: Num + std::ops::Mul<Output = T>>(scalar: T, v: &Vec<T>) -> Vec<T> {
    let mut r = vec![T::zero(); v.len()];
    for i in 0..v.len() {
        r[i] = v[i] * scalar;
    }

    return r;
}

/// Dense vector addition
///
/// # Example
/// ```
/// use gmres::dense_math::add_vec;
///
/// let a = vec![1, 2, 3];
///
/// let b = vec![1, 2, 3];
///
/// let c = add_vec(&a, &b, 1, 1);
///
/// assert_eq!(&c, &vec![2, 4, 6]);
/// ```
///
pub fn add_vec<T: Num>(a: &Vec<T>, b: &Vec<T>, alpha: T, beta: T) -> Vec<T>
where
    <T as std::ops::Mul>::Output: std::ops::Add<Output = T>,
{
    let len = a.len();
    if len != b.len() {
        panic!("Vector dimensions do not match.");
    }

    let mut c = Vec::with_capacity(len);
    for i in 0..len {
        c.push(alpha * a[i] + beta * b[i]);
    }
    return c;
}

/// Dense vector multiplication
///
/// # Example
/// ```
/// use gmres::dense_math::mul_vec;
///
/// let a = vec![1, 2, 3];
///
/// let b = vec![1, 2, 3];
///
/// let c = mul_vec(&a, &b);
///
/// assert_eq!(&c, &vec![1, 4, 9]);
/// ```
///
pub fn mul_vec<T: Num + std::ops::Mul<Output = T>>(a: &Vec<T>, b: &Vec<T>) -> Vec<T> {
    let len = a.len();
    if len != b.len() {
        panic!("Vector dimensions do not match.");
    }

    let mut c = Vec::with_capacity(len);
    for i in 0..len {
        c.push(a[i] * b[i]);
    }
    return c;
}

/// Dense vector dot product
///
/// ```
/// use gmres::dense_math::dot_prod;
///
/// let a = vec![0.9649, 0.1576, 0.9706];
///
/// let b = vec![0.9572, 0.4854, 0.8003];
///
/// let c = dot_prod(&a, &b);
///
/// assert!((c - 1.7768) < 1e-4);
/// ```
///
pub fn dot_prod<T: Num + std::ops::Mul<Output = T>>(a: &Vec<T>, b: &Vec<T>) -> T {
    let len = a.len();
    if len != b.len() {
        panic!("Vector dimensions do not match.");
    }

    let mut c = T::zero();
    for i in 0..len {
        c += a[i] * b[i];
    }

    return c;
}

/// Dense vector cross product
///
/// ```
/// use gmres::dense_math::cross_prod_3d;
///
/// let a = vec![0.9649, 0.1576, 0.9706];
///
/// let b = vec![0.9572, 0.4854, 0.8003];
///
/// let c = cross_prod_3d(&a, &b);
///
/// assert!((c[0] - -0.3450) < 1e-4);     
/// assert!((c[1] - 0.1568) < 1e-4);
/// assert!((c[2] - 0.3175) < 1e-4);
/// ```
///
pub fn cross_prod_3d<T: Num + std::ops::Mul<Output = T> + std::ops::Sub<Output = T>>(
    a: &Vec<T>,
    b: &Vec<T>,
) -> Vec<T> {
    let len = a.len();
    if len != b.len() {
        panic!("Vector dimensions do not match.");
    }
    if len != 3 {
        panic!("Vector dimension must be 3. {} given", len);
    }

    return vec![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
}

/// Dense matrix add
///
/// # Example
/// ```
/// use gmres::dense_math::add_mat;
///
/// let a = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
///
/// let b = vec![
///     vec![5, 5, 5],
///     vec![5, 5, 5],
///     vec![5, 5, 5],
/// ];
///
/// let c = add_mat(&a, &b, 1, 1);
///
/// assert_eq!(&c, &vec![
///     vec![6, 7, 8],
///     vec![9, 10, 11],
///     vec![12, 13, 14],
///     ]
/// );
/// ```
///
pub fn add_mat<T: Num + std::ops::Add<Output = T>>(
    a: &Vec<Vec<T>>,
    b: &Vec<Vec<T>>,
    alpha: T,
    beta: T,
) -> Vec<Vec<T>>
where
    <T as std::ops::Mul>::Output: std::ops::Add<Output = T>,
{
    // Check sizes
    let len_i = a.len();
    let len_j = a[0].len();
    if len_i != b.len() || len_j != b[0].len() {
        panic!("Matrix dimensions do not match.");
    }

    // Add
    let mut o = Vec::new();
    for i in 0..len_i {
        let mut row = Vec::new();
        for j in 0..len_j {
            row.push(alpha * a[i][j] + beta * b[i][j]);
        }
        o.push(row);
    }

    return o;
}

/// Dense matrix multiplication
///
/// # Example
/// ```
/// use gmres::dense_math::mul_mat;
///
/// let a = vec![
///     vec![1, 2, 3],
///     vec![4, 5, 6],
///     vec![7, 8, 9],
/// ];
///
/// let b = vec![
///     vec![5, 5, 5],
///     vec![5, 5, 5],
///     vec![5, 5, 5],
/// ];
///
/// let c = mul_mat(&a, &b);
///
/// assert_eq!(&c, &vec![
///     vec![30, 30, 30],
///     vec![75, 75, 75],
///     vec![120, 120, 120],
///     ]
/// );
/// ```
///
pub fn mul_mat<T: Num + std::ops::AddAssign<<T as std::ops::Mul>::Output>>(
    a: &Vec<Vec<T>>,
    b: &Vec<Vec<T>>,
) -> Vec<Vec<T>> {
    let rows_a = a.len();
    let columns_a = a[0].len();
    let rows_b = b.len();
    let columns_b = b[0].len();

    if columns_a != rows_b {
        panic!("Matrix dimensions do not match.")
    }

    let mut r = vec![vec![T::zero(); columns_b]; rows_a];
    for i in 0..rows_a {
        for j in 0..columns_b {
            for k in 0..rows_b {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }

    return r;
}

/// Transpose a dense matrix
///
/// # Example:
/// ```
/// use gmres::dense_math::transpose;
///
/// let a = vec![
///    vec![2.1615, 2.0044, 2.1312, 0.8217, 2.2074],
///    vec![2.2828, 1.9089, 1.9295, 0.9412, 2.0017],
///    vec![2.2156, 1.8776, 1.9473, 1.0190, 1.8352],
///    vec![1.0244, 0.8742, 0.9177, 0.7036, 0.7551],
///    vec![2.0367, 1.5642, 1.4313, 0.8668, 1.7571],
/// ];
///
/// assert_eq!(
///     transpose(&a),
///     vec![
///         vec![2.1615, 2.2828, 2.2156, 1.0244, 2.0367],
///         vec![2.0044, 1.9089, 1.8776, 0.8742, 1.5642],
///         vec![2.1312, 1.9295, 1.9473, 0.9177, 1.4313],
///         vec![0.8217, 0.9412, 1.0190, 0.7036, 0.8668],
///         vec![2.2074, 2.0017, 1.8352, 0.7551, 1.7571]
///      ]
///  );
/// ```
///
pub fn transpose<T: Num>(a: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    let rows = a.len();
    let columns = a[0].len();

    let mut r = vec![vec![T::zero(); rows]; columns];
    for i in 0..rows {
        for j in 0..columns {
            r[j][i] = a[i][j];
        }
    }

    return r;
}
