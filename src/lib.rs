pub mod layouts;
mod bit_utils;
mod utils;

use std::fmt;
use rayon::prelude::*;
use num_traits::{Zero, One};
use rand::Rng;
use rand::distributions::{Distribution, Uniform, Standard};

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

impl<T, L> fmt::Display for Array<T, L>
where
    T: Copy + Default + fmt::Display,
    L: Layout,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // Helper function to determine the prefix for each line based on depth
        fn get_indent(depth: usize) -> String {
            " ".repeat(depth)
        }

        fn fmt_recursive<T: fmt::Display + std::marker::Copy + std::default::Default>(
            f: &mut fmt::Formatter,
            array: &Array<T, impl Layout>,
            indices: &mut Vec<usize>,
            depth: usize,
        ) -> fmt::Result {
            if depth == array.dimensions {
                // We are at an individual element
                write!(f, "{}", array.get(indices).unwrap_or(&T::default()))?;
            } else {
                // We are at a dimension that needs to be iterated
                write!(f, "[")?;

                for i in 0..array.shape[depth] {
                    indices.push(i);

                    fmt_recursive(f, array, indices, depth + 1)?;

                    indices.pop();

                    let current_dim_size = array.shape[depth];
                    if current_dim_size > 0 && i < current_dim_size - 1 { // If not the last item at this level
                        let remaining_dims = array.dimensions - depth;
                        if remaining_dims <= 1 { // Separator for elements in 1D context (e.g. a row)
                            write!(f, " ")?;
                        } else if remaining_dims == 2 { // Separator for rows in 2D context
                            writeln!(f)?; // Newline between rows
                            write!(f, "{}", get_indent(depth + 1))?; // Indent for alignment
                        } else { // remaining_dims > 2. Separator for (N-1)D blocks in ND context (N >= 3)
                            writeln!(f)?; // First newline
                            writeln!(f)?; // Second newline (for separating 2D slices in 3D, etc.)
                            write!(f, "{}", get_indent(depth + 1))?; // Indent for alignment
                        }
                    }
                }
                write!(f, "]")?;
            }
            Ok(())
        }

        if self.data.is_empty() {
            return write!(f, "[]");
        }

        let mut indices = Vec::new();
        // For 0-dimensional arrays (scalars), just print the value.
        if self.dimensions == 0 {
            if let Some(value) = self.data.first() {
                return write!(f, "{}", value);
            } else {
                return write!(f, "{}", T::default()); // Or handle error appropriately
            }
        }
        fmt_recursive(f, self, &mut indices, 0)
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

        let array_f64 = Array::<f64, RowMajorOrderLayout>::default(vec![1, 3]);
        assert_eq!(array_f64.shape, vec![1, 3]);
        assert_eq!(array_f64.data, vec![0.0, 0.0, 0.0]); // Default for f64 is 0.0

        // Test with custom struct that implements Default
        #[derive(Copy, Clone, Debug, PartialEq, Default)]
        struct MyStruct { val: i32 }
        let array_custom = Array::<MyStruct, RowMajorOrderLayout>::default(vec![2]);
        assert_eq!(array_custom.shape, vec![2]);
        assert_eq!(array_custom.data, vec![MyStruct{val: 0}, MyStruct{val: 0}]);

        let scalar_array = Array::<i32, RowMajorOrderLayout>::default(vec![]);
        assert_eq!(scalar_array.shape, Vec::<usize>::new());
        assert_eq!(scalar_array.data, vec![0]);
    }
}

