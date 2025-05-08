pub mod layouts;
mod bit_utils;
mod utils;

use rayon::prelude::*;

pub trait Layout {
    fn new(shape: Vec<usize>) -> Self;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}

#[derive(Debug)]
pub struct Array<T, L: Layout>
where
    T: Copy + Default,
{
    data: Vec<T>,
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

