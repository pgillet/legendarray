use legendarray::layouts::RowMajorOrderLayout;
use legendarray::Array;

fn main() {
    println!("--- Array Creation Examples ---");

    // 1. Array::default()
    println!("\n1. Array created with default():");
    let arr_default_2d = Array::<i32, RowMajorOrderLayout>::default(vec![2, 3]);
    println!("2D i32 default (should be all zeros):\n{}", arr_default_2d);
    let arr_default_scalar = Array::<f64, RowMajorOrderLayout>::default(vec![]);
    println!("Scalar f64 default (should be 0.0):\n{}", arr_default_scalar);

    // 2. Array::zeros()
    println!("\n2. Array created with zeros():");
    let arr_zeros_3d = Array::<f32, RowMajorOrderLayout>::zeros(vec![2, 2, 2]);
    println!("3D f32 zeros:\n{}", arr_zeros_3d);
    let arr_zeros_scalar = Array::<u8, RowMajorOrderLayout>::zeros(vec![]);
    println!("Scalar u8 zeros:\n{}", arr_zeros_scalar);

    // 3. Array::ones()
    println!("\n3. Array created with ones():");
    let arr_ones_1d = Array::<i64, RowMajorOrderLayout>::ones(vec![4]);
    println!("1D i64 ones:\n{}", arr_ones_1d);
    let arr_ones_scalar = Array::<i32, RowMajorOrderLayout>::ones(vec![]);
    println!("Scalar i32 ones:\n{}", arr_ones_scalar);

    // 4. Array::arange_sequential()
    println!("\n4. Array created with arange_sequential():");
    let arr_arange_2d = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![3, 3]);
    println!("2D i32 arange_sequential:\n{}", arr_arange_2d);
    let arr_arange_scalar = Array::<u32, RowMajorOrderLayout>::arange_sequential(vec![]);
    println!("Scalar u32 arange_sequential (should be 0):\n{}", arr_arange_scalar);

    // 5. Array::random_uniform()
    println!("\n5. Array created with random_uniform():");
    let arr_rand_uniform_2d = Array::<f64, RowMajorOrderLayout>::random_uniform(vec![2, 4], -5.0, 5.0);
    println!("2D f64 random_uniform [-5.0, 5.0]:\n{}", arr_rand_uniform_2d);
    let arr_rand_uniform_scalar = Array::<f32, RowMajorOrderLayout>::random_uniform(vec![], 100.0, 101.0);
    println!("Scalar f32 random_uniform [100.0, 101.0]:\n{}", arr_rand_uniform_scalar);

    // 6. Array::random_standard()
    println!("\n6. Array created with random_standard():");
    let arr_rand_std_2d = Array::<f32, RowMajorOrderLayout>::random_standard(vec![2, 2]);
    println!("2D f32 random_standard (usually [0.0, 1.0) for floats):\n{}", arr_rand_std_2d);
    let arr_rand_std_bool_1d = Array::<bool, RowMajorOrderLayout>::random_standard(vec![5]);
    println!("1D bool random_standard:\n{}", arr_rand_std_bool_1d);
    let arr_rand_std_scalar = Array::<f64, RowMajorOrderLayout>::random_standard(vec![]);
    println!("Scalar f64 random_standard:\n{}", arr_rand_std_scalar);

    println!("\n--- End of Array Creation Examples ---");
} 