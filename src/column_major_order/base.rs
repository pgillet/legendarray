use crate::Layout;

struct ColumnMajorOrderLayout {
    shape: Vec<usize>,
}

impl Layout for ColumnMajorOrderLayout {
    fn new(shape: Vec<usize>) -> Self {
        ColumnMajorOrderLayout {
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
        for (dim_size, &dim_index) in self.shape.iter().zip(indices.iter()) {
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
    fn test_index_1d() {
        let shape = vec![5];
        let indices = vec![2];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(2));
    }

    #[test]
    fn test_index_2d() {
        let shape = vec![3, 4];
        let indices = vec![1, 2];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(7));
    }

    #[test]
    fn test_index_2d_bis() {
        let shape = vec![4, 3];
        let indices = vec![1, 2];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(9));
    }

    #[test]
    fn test_index_3d() {
        let shape = vec![3, 4, 5];
        let indices = vec![2, 3, 4];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(59));
    }

    #[test]
    fn test_index_3d_bis() {
        let shape = vec![3, 4, 5];
        let indices = vec![1, 2, 2];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(31));
    }

    #[test]
    fn test_index_4d() {
        let shape = vec![2, 3, 4, 5];
        let indices = vec![1, 2, 3, 4];
        let layout = ColumnMajorOrderLayout::new(shape);
        assert_eq!(layout.index(&indices), Some(119));
    }

    #[test]
    #[should_panic(expected = "The number of indices does not match the number of dimensions.")]
    fn test_index_mismatch_dimensions() {
        let shape = vec![3, 4];
        let indices = vec![1, 2, 3];
        let layout = ColumnMajorOrderLayout::new(shape);
        layout.index(&indices);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds.")]
    fn test_index_index_out_of_bounds() {
        let shape = vec![3, 4];
        let indices = vec![3, 4];
        let layout = ColumnMajorOrderLayout::new(shape);
        layout.index(&indices);
    }
}
