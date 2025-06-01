use legendarray::layouts::ColumnMajorOrderLayout;
use legendarray::Array;
use legendarray::layouts::row_major::RowMajorOrderLayout;

// Example usage
fn main() {

    let array_1d = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![6]);
    println!("{}", array_1d);

    // 2D array
    let shape_2d = vec![4, 3];
    let array_2d = Array::<i32, ColumnMajorOrderLayout>::arange_sequential(shape_2d);
    println!("\n2D Array:");
    println!("{}", array_2d);

    // 3D array
    let shape_3d = vec![2, 3, 4];
    let array_3d = Array::<i32, RowMajorOrderLayout>::arange_sequential(shape_3d);
    println!("\n3D Array:");
    println!("{}", array_3d);


    // 4D array
    let shape_4d = vec![2, 3, 4, 5];
    let array_4d = Array::<i32, RowMajorOrderLayout>::arange_sequential(shape_4d);
    println!("\n4D Array:");
    println!("{}", array_4d);
}