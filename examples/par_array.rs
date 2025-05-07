use legendarray::Array;
use legendarray::layouts::row_major::RowMajorOrderLayout;

// Example usage
fn main() {
    let mut array = Array::<i32, RowMajorOrderLayout>::new(vec![3, 4, 2]);

    // Parallel map: Double each element
    let doubled = array.par_map(|&x| x * 2);

    // Parallel apply: Add 5 to each element
    array.par_apply(|x| *x += 5);

    // Get a 2D slice (fixed first dimension = 1) and sum its elements
    let slice_sum = array.par_slice(0, 1, |&x| x)
        .map(|slice| slice.iter().sum::<i32>())
        .unwrap_or(0);
}