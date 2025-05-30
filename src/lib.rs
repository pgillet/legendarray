pub mod layouts;
mod bit_utils;
mod utils;

use std::fmt;
use rayon::prelude::*;

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
    pub fn new(shape: Vec<usize>) -> Self {
        let dimensions = shape.len();
        let size: usize = shape.iter().product();
        let data = vec![T::default(); size]; // Initialize with default values

        let layout = L::new(shape.clone());
        Self {
            data,
            shape,
            dimensions,
            layout,
        }
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

    // pub fn range(start: A, end: A, step: A) -> Self
    // where A: Float
    // {
    //     Self::from(to_vec(linspace::range(start, end, step)))
    // }


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
        let mut array = Array::<i32, RowMajorOrderLayout>::new(vec![2, 3]);
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
        let mut array = Array::<i32, RowMajorOrderLayout>::new(vec![2, 3]);
        let data = vec![1, 2, 3, 4, 5, 6];
        array.data = data; // Directly set the data for testing purposes

        // Perform par_apply operation
        array.par_apply(|x| *x += 10);

        // Verify the result
        assert_eq!(array.data, vec![11, 12, 13, 14, 15, 16]);
    }
}

