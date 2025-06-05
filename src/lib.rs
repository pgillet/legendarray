pub mod layouts;
mod bit_utils;
mod utils;

use std::fmt;
use rayon::prelude::*;
use num_traits::{Zero, One};
use rand::Rng;
use rand::distributions::{Distribution, Uniform, Standard};

// Add these new types for print options
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrintLayoutOrder {
    DefaultVisualOrder, // Standard layout
    TransposeFirstTwoAxes, // Transposes the first two logical axes for display
}

#[derive(Clone, Copy, Debug)]
pub struct PrintOptions {
    pub layout_order: PrintLayoutOrder,
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self { layout_order: PrintLayoutOrder::DefaultVisualOrder }
    }
}
// End of new types for print options

pub trait Layout: Sync {
    fn new(shape: Vec<usize>) -> Self;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}

#[derive(Debug)]
pub struct Array<T, L: Layout>
where
    T: Copy + Default,
{
    pub data: Vec<T>,
    shape: Vec<usize>,
    dimensions: usize,
    layout: L,
}

// Add this new public struct for custom printing
pub struct ArrayPrinter<'a, T, L: Layout>
where
    T: Copy + Default + fmt::Display,
{
    array: &'a Array<T, L>,
    options: PrintOptions,
}
// End of ArrayPrinter struct definition

impl<T, L: Layout + std::marker::Sync> Array<T, L>
where
    T: Copy + Default,
{
    fn new<F>(shape: Vec<usize>, mut generator: F) -> Self
    where
        F: FnMut() -> T,
    {
        let dimensions = shape.len();
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            data.push(generator());
        }

        let layout = L::new(shape.clone());
        Self {
            data,
            shape,
            dimensions,
            layout,
        }
    }

    pub fn default(shape: Vec<usize>) -> Self {
        Self::new(shape, || T::default())
    }

    pub fn zeros(shape: Vec<usize>) -> Self
    where T: Zero
    {
        Self::new(shape, || T::zero())
    }

    pub fn ones(shape: Vec<usize>) -> Self
    where T: One
    {
        Self::new(shape, || T::one())
    }

    /// Creates an array filled with sequential values starting from T::zero().
    /// E.g., 0, 1, 2, ...
    pub fn arange_sequential(shape: Vec<usize>) -> Self
    where T: Zero + One + std::ops::AddAssign<T> + Copy
    {
        let mut current_val = T::zero();
        Self::new(shape, || {
            let val_to_return = current_val;
            current_val += T::one();
            val_to_return
        })
    }

    /// Creates an array with random values sampled uniformly from [low, high].
    pub fn random_uniform(shape: Vec<usize>, low: T, high: T) -> Self
    where
        T: rand::distributions::uniform::SampleUniform + PartialOrd,
    {
        let mut rng = rand::thread_rng();
        // Ensure low <= high for Uniform::new_inclusive
        let (actual_low, actual_high) = if low <= high { (low, high) } else { (high, low) };
        let dist = Uniform::new_inclusive(actual_low, actual_high);
        Self::new(shape, || dist.sample(&mut rng))
    }

    /// Creates an array with random values using the Standard distribution.
    /// For many types, this produces values in a standard range (e.g., floats in [0,1)).
    pub fn random_standard(shape: Vec<usize>) -> Self
    where
        Standard: Distribution<T>,
    {
        let mut rng = rand::thread_rng();
        Self::new(shape, || rng.gen())
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let index = self.layout.index(indices)?;
        self.data.get(index)
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let index = self.layout.index(indices)?;
        self.data.get_mut(index)
    }

    pub fn set(&mut self, indices: &[usize], value: T) -> Option<()> {
        let index = self.layout.index(indices)?;
        if index >= self.data.len() {
            self.data.resize(index + 1, T::default());
        }
        self.data[index] = value;
        Some(())
    }

    // Map a function over all elements in parallel
    pub fn par_map<F, R>(&self, f: F) -> Vec<R>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send,
    {
        self.data.par_iter().map(f).collect()
    }

    // Apply a function to all elements in parallel and update them in-place
    pub fn par_apply<F>(&mut self, f: F)
    where
        T: Send + Sync,
        F: Fn(&mut T) + Sync + Send,
    {
        self.data.par_iter_mut().for_each(f);
    }

    // Get a slice of data in parallel
    pub fn par_slice<F, R>(&self, dim: usize, idx: usize, f: F) -> Option<Vec<R>>
    where
        T: Sync,
        R: Send,
        F: Fn(&T) -> R + Sync + Send,
        L: Layout + Sync,
    {
        if dim >= self.shape.len() || idx >= self.shape[dim] {
            return None;
        }

        // Create indices for the slice
        let mut slice_indices = Vec::new();
        let mut current = vec![0; self.shape.len()];
        current[dim] = idx;

        // Calculate total elements in the slice
        let slice_size: usize = self.shape.iter()
            .enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .product();

        slice_indices.reserve(slice_size);

        for _ in 0..slice_size {
            slice_indices.push(current.clone());

            // Update indices, skipping the fixed dimension
            for d in (0..self.shape.len()).rev() {
                if d == dim {
                    continue;
                }

                current[d] += 1;
                if current[d] < self.shape[d] {
                    break;
                }
                current[d] = 0;
            }
        }

        // Process the slice in parallel
        Some(slice_indices.into_par_iter()
            .filter_map(|idx| self.layout.index(&idx).map(|i| &self.data[i]))
            .map(f)
            .collect())
    }
}

// New impl block for display_with_options with explicit L: Layout
impl<T, L: Layout> Array<T, L> // Explicit L: Layout here
where
    T: Copy + Default + fmt::Display, // Ensure T can be displayed
{
    pub fn display_with_options(&self, options: PrintOptions) -> ArrayPrinter<'_, T, L> {
        ArrayPrinter { array: self, options }
    }
}

// Modify the existing Display implementation for Array
impl<T, L: Layout> fmt::Display for Array<T, L> // L: Layout is correctly here
where
    T: Copy + Default + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.display_with_options(PrintOptions::default()).fmt(f)
    }
}

// Implement Display for ArrayPrinter
impl<'a, T, L: Layout> fmt::Display for ArrayPrinter<'a, T, L>
where
    T: Copy + Default + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let array_dims = self.array.dimensions;

        if array_dims == 0 {
            return write!(f, "{}", self.array.data.first().unwrap_or(&T::default()));
        }
        if self.array.data.is_empty() && array_dims > 0 {
             // Constructing [] for empty array with non-zero dimensions.
            // For example, shape [2,0,3] should print as [[] []] or similar if we were to fully represent.
            // Simpler to just print [] if data is empty but dimensions > 0.
            // However, arange_sequential for [2,0,3] creates empty data and thus prints [].
            // If shape is [0], data is [val], dimensions = 1. (This is not typical, shape usually non-empty for dim > 0)
            // A shape like [0] is usually disallowed by constructors or means size 0.
            // If shape is vec![], then dimensions = 0, handled above.
            // The current test for [2,0,3] expects "[]", so this is fine.
            return write!(f, "[]");
        }

        let mut axis_permutation: Vec<usize> = (0..array_dims).collect();
        if self.options.layout_order == PrintLayoutOrder::TransposeFirstTwoAxes && array_dims >= 2 {
            axis_permutation.swap(0, 1);
        }

        let mut logical_indices_for_get = vec![0; array_dims];
        
        fn get_indent(visual_depth: usize) -> String {
            " ".repeat(visual_depth) // Indent based on visual nesting of brackets
        }

        fn fmt_recursive_printer<TFmt, LFmt: Layout>(
            f_rec: &mut fmt::Formatter,
            target_array: &Array<TFmt, LFmt>,
            current_axis_permutation: &[usize],
            current_logical_indices: &mut [usize], // Use slice for interior mutability without realloc
            current_visual_depth: usize,
        ) -> fmt::Result
        where
            TFmt: Copy + Default + fmt::Display,
        {
            let num_array_dims = target_array.dimensions;

            if current_visual_depth == num_array_dims { // Base case: print element
                write!(f_rec, "{}", target_array.get(current_logical_indices).unwrap_or(&TFmt::default()))?
            } else {
                write!(f_rec, "[")?;

                let logical_dim_to_iterate = current_axis_permutation[current_visual_depth];
                let dim_size = target_array.shape[logical_dim_to_iterate];

                if dim_size > 0 { // Only iterate if dimension is not zero-sized
                    let mut peekable_iter = (0..dim_size).peekable();
                    while let Some(val_in_current_logical_dim) = peekable_iter.next() {
                        current_logical_indices[logical_dim_to_iterate] = val_in_current_logical_dim;
                        
                        fmt_recursive_printer(
                            f_rec, 
                            target_array, 
                            current_axis_permutation, 
                            current_logical_indices, 
                            current_visual_depth + 1
                        )?;
                        
                        if peekable_iter.peek().is_some() { // If not the last element for this visual depth
                            // Separator logic based on the visual role of the current_visual_depth
                            let visual_remaining_depth = num_array_dims - (current_visual_depth + 1); 
                            // +1 because current_visual_depth is 0 for outermost, N-1 for innermost element list

                            if visual_remaining_depth == 0 { // Innermost elements on a line (visual_depth == N-1)
                                write!(f_rec, " ")?;
                            } else if visual_remaining_depth == 1 { // Rows (visual_depth == N-2)
                                writeln!(f_rec)?;
                                // Indent for the next line is based on how many brackets are open for *that next line*.
                                // The items of the current visual_depth are being printed. Indent for next line is visual_depth + 1 for its items.
                                write!(f_rec, "{}", get_indent(current_visual_depth + 1))?;
                            } else { // Blocks (visual_depth < N-2)
                                writeln!(f_rec)?;
                                writeln!(f_rec)?;
                                write!(f_rec, "{}", get_indent(current_visual_depth + 1))?;
                            }
                        }
                    }
                } else {
                    // Dimension is zero-sized. Handled by outer is_empty check or prints nothing inside brackets.
                    // If shape is [2,0,3], data is empty. Prints [].
                    // If we allow non-empty data with zero-sized dimension (not typical), this prints [].
                }
                write!(f_rec, "]")?;
            }
            Ok(())
        }
        fmt_recursive_printer(f, self.array, &axis_permutation, &mut logical_indices_for_get, 0)
    }
}

#[cfg(test)]
mod tests {
    use crate::layouts::RowMajorOrderLayout;
    use super::*;

    #[test]
    fn test_par_map() {
        // Create an Array with RowMajorOrderLayout and initialize with data
        let mut array = Array::<i32, RowMajorOrderLayout>::default(vec![2, 3]);
        let data = vec![1, 2, 3, 4, 5, 6];
        array.data = data; // Directly set the data for testing purposes

        // Perform par_map operation
        let result = array.par_map(|&x| x * 2);

        // Verify the result
        assert_eq!(result, vec![2, 4, 6, 8, 10, 12]);
    }

    #[test]
    fn test_par_apply() {
        // Create an Array with RowMajorOrderLayout and initialize with data
        let mut array = Array::<i32, RowMajorOrderLayout>::default(vec![2, 3]);
        let data = vec![1, 2, 3, 4, 5, 6];
        array.data = data; // Directly set the data for testing purposes

        // Perform par_apply operation
        array.par_apply(|x| *x += 10);

        // Verify the result
        assert_eq!(array.data, vec![11, 12, 13, 14, 15, 16]);
    }

    #[test]
    fn test_zeros_constructor() {
        let array = Array::<f64, RowMajorOrderLayout>::zeros(vec![2, 2]);
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data, vec![0.0, 0.0, 0.0, 0.0]);
        let scalar_array = Array::<i32, RowMajorOrderLayout>::zeros(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data, vec![0]);
    }

    #[test]
    fn test_ones_constructor() {
        let array = Array::<f32, RowMajorOrderLayout>::ones(vec![1, 3]);
        assert_eq!(array.shape, vec![1, 3]);
        assert_eq!(array.data, vec![1.0, 1.0, 1.0]);
        let scalar_array = Array::<i32, RowMajorOrderLayout>::ones(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data, vec![1]);
    }

    #[test]
    fn test_arange_sequential_constructor() {
        let array = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![2, 3]);
        assert_eq!(array.shape, vec![2, 3]);
        assert_eq!(array.data, vec![0, 1, 2, 3, 4, 5]);
        let array_f64 = Array::<f64, RowMajorOrderLayout>::arange_sequential(vec![3]);
        assert_eq!(array_f64.shape, vec![3]);
        assert_eq!(array_f64.data, vec![0.0, 1.0, 2.0]);
        let scalar_array = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data, vec![0]);
    }

    #[test]
    fn test_random_uniform_constructor() {
        let array = Array::<f32, RowMajorOrderLayout>::random_uniform(vec![2, 2], 0.0, 10.0);
        assert_eq!(array.shape, vec![2, 2]);
        assert_eq!(array.data.len(), 4);
        for &val in &array.data {
            assert!(val >= 0.0 && val <= 10.0);
        }

        // Test with inverted bounds
        let array_inv = Array::<f32, RowMajorOrderLayout>::random_uniform(vec![1, 2], 10.0, 0.0);
        assert_eq!(array_inv.shape, vec![1, 2]);
        assert_eq!(array_inv.data.len(), 2);
        for &val in &array_inv.data {
            assert!(val >= 0.0 && val <= 10.0);
        }

        let scalar_array = Array::<f64, RowMajorOrderLayout>::random_uniform(vec![], -1.0, 1.0);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data.len(), 1);
        assert!(scalar_array.data[0] >= -1.0 && scalar_array.data[0] <= 1.0);
    }

    #[test]
    fn test_random_standard_constructor() {
        let array_f64 = Array::<f64, RowMajorOrderLayout>::random_standard(vec![2, 1]);
        assert_eq!(array_f64.shape, vec![2, 1]);
        assert_eq!(array_f64.data.len(), 2);
        for &val in &array_f64.data {
            // Standard distribution for f64 is typically [0.0, 1.0)
            assert!(val >= 0.0 && val < 1.0);
        }

        let array_bool = Array::<bool, RowMajorOrderLayout>::random_standard(vec![5]);
        assert_eq!(array_bool.shape, vec![5]);
        assert_eq!(array_bool.data.len(), 5);
        // Values will be true or false

        let scalar_array = Array::<f32, RowMajorOrderLayout>::random_standard(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data.len(), 1);
        assert!(scalar_array.data[0] >= 0.0 && scalar_array.data[0] < 1.0);
    }

    #[test]
    fn test_new_constructor_with_default() {
        // Test Array::new which uses T::default()
        let array_i32 = Array::<i32, RowMajorOrderLayout>::default(vec![2, 2]);
        assert_eq!(array_i32.shape, vec![2, 2]);
        assert_eq!(array_i32.data, vec![0, 0, 0, 0]); // Default for i32 is 0
        assert_eq!(array_i32.to_string(), "[[0 0]\n [0 0]]");

        let array_f64 = Array::<f64, RowMajorOrderLayout>::default(vec![1, 3]);
        assert_eq!(array_f64.shape, vec![1, 3]);
        assert_eq!(array_f64.data, vec![0.0, 0.0, 0.0]); // Default for f64 is 0.0
        assert_eq!(array_f64.to_string(), "[[0 0 0]]");

        // Test with custom struct that implements Default
        #[derive(Copy, Clone, Debug, PartialEq, Default)]
        struct MyStruct { val: i32 }
        impl fmt::Display for MyStruct { // Ensure MyStruct is displayable for tests
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.val)
            }
        }
        let array_custom = Array::<MyStruct, RowMajorOrderLayout>::default(vec![2]);
        assert_eq!(array_custom.shape, vec![2]);
        assert_eq!(array_custom.data, vec![MyStruct{val: 0}, MyStruct{val: 0}]);
        assert_eq!(array_custom.to_string(), "[0 0]");

        let scalar_array = Array::<i32, RowMajorOrderLayout>::default(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data, vec![0]);
        assert_eq!(scalar_array.to_string(), "0");
    }

    // Add new tests for print options
    #[test]
    fn test_display_with_options_normal_order() {
        let array = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![2,3]);
        // Default options (Normal order)
        // [[0 1 2]
        //  [3 4 5]]
        let expected_normal = "[[0 1 2]\n [3 4 5]]";
        assert_eq!(array.to_string(), expected_normal);
        assert_eq!(array.display_with_options(PrintOptions::default()).to_string(), expected_normal);
    }

    // Renaming test and updating expected output for TransposeFirstTwoAxes
    #[test]
    fn test_display_with_options_transposed_axes_order() {
        let array_2x3 = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![2,3]);
        // Data: [0,1,2,3,4,5]. Shape [2,3]. get([r,c])
        // Default: [[0 1 2]
        //           [3 4 5]]
        // Transposed (visual swap of logical axes 0 and 1):
        // Visual rows iterate logical dim 1 (0..3). Visual cols iterate logical dim 0 (0..2).
        // Printed value at (v_row, v_col) is array.get([v_col, v_row])
        // v_row=0 (log1=0): v_col=0 (log0=0) -> get([0,0])=0. v_col=1 (log0=1) -> get([1,0])=3.
        // v_row=1 (log1=1): v_col=0 (log0=0) -> get([0,1])=1. v_col=1 (log0=1) -> get([1,1])=4.
        // v_row=2 (log1=2): v_col=0 (log0=0) -> get([0,2])=2. v_col=1 (log0=1) -> get([1,2])=5.
        // Expected: [[0 3]
        //            [1 4]
        //            [2 5]]
        let options_transposed = PrintOptions { layout_order: PrintLayoutOrder::TransposeFirstTwoAxes };
        let expected_transposed_2x3 = "[[0 3]\n [1 4]\n [2 5]]";
        assert_eq!(array_2x3.display_with_options(options_transposed).to_string(), expected_transposed_2x3);

        let array_1d = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![4]); // data [0,1,2,3]
        // TransposeFirstTwoAxes has no effect on 1D or 0D, should be same as default.
        assert_eq!(array_1d.display_with_options(options_transposed).to_string(), "[0 1 2 3]");
        assert_eq!(array_1d.to_string(), "[0 1 2 3]");

        let scalar_array = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![]); // data [0]
        assert_eq!(scalar_array.display_with_options(options_transposed).to_string(), "0");
        assert_eq!(scalar_array.to_string(), "0");

        // Test with a 3D array
        let array_2x2x2 = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![2,2,2]);
        // Data: [0,1,2,3,4,5,6,7]
        // Default: [[[0 1]
        //            [2 3]]
        //
        //           [[4 5]
        //            [6 7]]]
        // Transposed (swap visual roles of logical dim 0 and 1):
        // axis_permutation = [1,0,2]
        // Outermost visual loop (visual_depth=0) iterates logical dim 1 (shape[1]=2 elements).
        // Middle visual loop    (visual_depth=1) iterates logical dim 0 (shape[0]=2 elements).
        // Innermost visual loop (visual_depth=2) iterates logical dim 2 (shape[2]=2 elements).
        // Separators: vd=2 space, vd=1 newline, vd=0 double newline.
        // vd=0, log_dim_idx=1, val_log1=0: (indices_for_get[1]=0)
        //   vd=1, log_dim_idx=0, val_log0=0: (indices_for_get[0]=0)
        //     vd=2, log_dim_idx=2, val_log2=0: (indices_for_get[2]=0) -> get([0,0,0])=0. Space.
        //     vd=2, log_dim_idx=2, val_log2=1: (indices_for_get[2]=1) -> get([0,0,1])=1.
        //   Newline. vd=1, log_dim_idx=0, val_log0=1: (indices_for_get[0]=1)
        //     vd=2, log_dim_idx=2, val_log2=0: (indices_for_get[2]=0) -> get([1,0,0])=2. Space.
        //     vd=2, log_dim_idx=2, val_log2=1: (indices_for_get[2]=1) -> get([1,0,1])=3.
        // Double newline. vd=0, log_dim_idx=1, val_log1=1: (indices_for_get[1]=1)
        //   vd=1, log_dim_idx=0, val_log0=0: (indices_for_get[0]=0)
        //     vd=2, log_dim_idx=2, val_log2=0: (indices_for_get[2]=0) -> get([0,1,0])=4. Space.
        //     vd=2, log_dim_idx=2, val_log2=1: (indices_for_get[2]=1) -> get([0,1,1])=5.
        //   Newline. vd=1, log_dim_idx=0, val_log0=1: (indices_for_get[0]=1)
        //     vd=2, log_dim_idx=2, val_log2=0: (indices_for_get[2]=0) -> get([1,1,0])=6. Space.
        //     vd=2, log_dim_idx=2, val_log2=1: (indices_for_get[2]=1) -> get([1,1,1])=7.
        // Expected: [[[0 1]
        //             [2 3]]
        // 
        //            [[4 5]
        //             [6 7]]]  Wait, this is the same as default for this case. This is because get([0,0,0]) etc. are still called.
        // The permutation in axis_permutation maps visual depth to logical dimension iterated.
        // The logical_indices_for_get has its elements at [logical_dim_idx] set.
        // This is correct. The visual output for this specific 3D example might look similar if the values align.
        // Let's write it out based on test array_2x3 which has distinct numbers after transpose.
        // Expected Transposed 2x2x2:
        // Slice 0 (iterated by val_log1=0 for vd=0):
        //   Row 0 (iterated by val_log0=0 for vd=1):
        //     Item 0 (iterated by val_log2=0 for vd=2): get([0,0,0]) -> 0
        //     Item 1 (iterated by val_log2=1 for vd=2): get([0,0,1]) -> 1
        //   Row 1 (iterated by val_log0=1 for vd=1):
        //     Item 0 (iterated by val_log2=0 for vd=2): get([1,0,0]) -> 2
        //     Item 1 (iterated by val_log2=1 for vd=2): get([1,0,1]) -> 3
        // Slice 1 (iterated by val_log1=1 for vd=0):
        //   Row 0 (iterated by val_log0=0 for vd=1):
        //     Item 0 (iterated by val_log2=0 for vd=2): get([0,1,0]) -> 4
        //     Item 1 (iterated by val_log2=1 for vd=2): get([0,1,1]) -> 5
        //   Row 1 (iterated by val_log0=1 for vd=1):
        //     Item 0 (iterated by val_log2=0 for vd=2): get([1,1,0]) -> 6
        //     Item 1 (iterated by val_log2=1 for vd=2): get([1,1,1]) -> 7
        // This structure is exactly the same as the default print because the values are just permuted in terms of what logical dim they came from.
        // The visual structure (brackets and newlines) is driven by visual_depth. The values are from get(logical_indices).
        // So for 2x2x2, default and transposed will print the numbers in the same visual spots.
        // This is because swapping logical dims 0 and 1, when they have the same size (2), means permutation [1,0,2] will iterate shape[1] then shape[0] then shape[2].
        // If shape[0]=A, shape[1]=B. Default iterates A then B. Transposed iterates B then A.
        // Test with shape [2,3,2]
        let array_2x3x2 = Array::<i32, RowMajorOrderLayout>::arange_sequential(vec![2,3,2]);
        // Data size 12: [0..11]
        // Default print for [2,3,2]:
        // [[[0 1]
        //   [2 3]
        //   [4 5]]
        // 
        //  [[6 7]
        //   [8 9]
        //   [10 11]]]
        let expected_default_2x3x2 = "[[[0 1]\n  [2 3]\n  [4 5]]\n\n [[6 7]\n  [8 9]\n  [10 11]]]";
        assert_eq!(array_2x3x2.to_string(), expected_default_2x3x2);
        // Transposed print for [2,3,2] (axis_perm = [1,0,2])
        // vd=0 iterates log_dim 1 (size 3). vd=1 iterates log_dim 0 (size 2). vd=2 iterates log_dim 2 (size 2).
        // Slice 0 (log_dim1_val=0):
        //   Row 0 (log_dim0_val=0):
        //     Items (log_dim2_val=0,1): get([0,0,0])=0, get([0,0,1])=1. -> [0 1]
        //   Row 1 (log_dim0_val=1):
        //     Items (log_dim2_val=0,1): get([1,0,0])=6, get([1,0,1])=7. -> [6 7]
        // Slice 1 (log_dim1_val=1):
        //   Row 0 (log_dim0_val=0):
        //     Items (log_dim2_val=0,1): get([0,1,0])=2, get([0,1,1])=3. -> [2 3]
        //   Row 1 (log_dim0_val=1):
        //     Items (log_dim2_val=0,1): get([1,1,0])=8, get([1,1,1])=9. -> [8 9]
        // Slice 2 (log_dim1_val=2):
        //   Row 0 (log_dim0_val=0):
        //     Items (log_dim2_val=0,1): get([0,2,0])=4, get([0,2,1])=5. -> [4 5]
        //   Row 1 (log_dim0_val=1):
        //     Items (log_dim2_val=0,1): get([1,2,0])=10, get([1,2,1])=11. -> [10 11]
        // Expected:
        // [[[0 1]
        //   [6 7]]
        // 
        //  [[2 3]
        //   [8 9]]
        // 
        //  [[4 5]
        //   [10 11]]]
        let expected_transposed_2x3x2 = "[[[0 1]\n  [6 7]]\n\n [[2 3]\n  [8 9]]\n\n [[4 5]\n  [10 11]]]";
        assert_eq!(array_2x3x2.display_with_options(options_transposed).to_string(), expected_transposed_2x3x2);
    }
}

