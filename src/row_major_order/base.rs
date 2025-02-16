use crate::Layout;

struct RowMajorOrderLayout {
    shape: Vec<usize>,
}

impl RowMajorOrderLayout {
    fn are_indices_valid(&self, indices: &[usize]) -> bool {
        // Check if the number of indices matches the number of dimensions
        if self.shape.len() != indices.len() {
            return false;
        }

        // Iterate through both shape and indices simultaneously
        for (dim_size, &index) in self.shape.iter().zip(indices.iter()) {
            // Check if the index is within the valid range
            if index >= *dim_size {
                return false;
            }
        }

        true
    }
}

impl Layout for RowMajorOrderLayout {
    fn new(shape: Vec<usize>) -> Self {
        RowMajorOrderLayout {
            shape,
        }
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {

        // Check if the number of indices matches the number of dimensions
        if indices.len() != self.shape.len() {
            panic!("The number of indices does not match the number of dimensions.");
        }

        // Initialize the index to 0
        let mut index = 0;
        let mut stride = 1;

        // Iterate through the dimensions in reverse order
        for (dim_size, &dim_index) in self.shape.iter().rev().zip(indices.iter().rev()) {
            // Check if the index is within the valid range
            if dim_index >= *dim_size {
                panic!("Index out of bounds.");
            }
            // Update the index
            index += dim_index * stride;
            // Update the stride for the next dimension
            stride *= dim_size;
        }

        Some(index)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_compute_address_1d() {
        let shape = vec![5];
        let indices = vec![2];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(2));
    }

    #[test]
    fn test_compute_address_2d() {
        let shape = vec![3, 4];
        let indices = vec![1, 2];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(6));
    }

    #[test]
    fn test_compute_address_2d_bis() {
        let shape = vec![3, 4];
        let indices = vec![2, 1];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(9));
    }

    #[test]
    fn test_compute_address_2d_ter() {
        let shape = vec![3, 3];
        let indices = vec![2, 2];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(8));
    }

    #[test]
    fn test_compute_address_3d() {
        let shape = vec![3, 4, 5];
        let indices = vec![2, 3, 4];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(59));
    }

    #[test]
    fn test_compute_address_3d_bis() {
        let shape = vec![3, 3, 3];
        let indices = vec![2, 1, 1];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(22));
    }

    #[test]
    fn test_compute_address_4d() {
        let shape = vec![2, 3, 4, 5];
        let indices = vec![1, 2, 3, 4];
        let layout = RowMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(119));
    }

    #[test]
    #[should_panic(expected = "The number of indices does not match the number of dimensions.")]
    fn test_compute_address_mismatch_dimensions() {
        let shape = vec![3, 4];
        let indices = vec![1, 2, 3];
        let layout = RowMajorOrderLayout::new(shape);
        layout.index(&indices);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds.")]
    fn test_compute_address_index_out_of_bounds() {
        let shape = vec![3, 4];
        let indices = vec![3, 4];
        let layout = RowMajorOrderLayout::new(shape);
        layout.index(&indices);
    }
}
