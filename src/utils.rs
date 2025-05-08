pub fn are_indices_valid(shape: &[usize], indices: &[usize]) -> bool {
    // Check if the number of indices matches the number of dimensions
    if shape.len() != indices.len() {
        return false;
    }

    // Iterate through both shape and indices simultaneously
    for (dim_size, &index) in shape.iter().zip(indices.iter()) {
        // Check if the index is within the valid range
        if index >= *dim_size {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::are_indices_valid;

    #[test]
    fn test_are_indices_valid() {
        // Test case 1: Valid indices
        let shape = vec![3, 4, 2];
        let indices = vec![1, 2, 0];
        assert!(are_indices_valid(&shape, &indices));

        // Test case 2: Indices out of bounds
        let shape = vec![3, 4, 2];
        let indices = vec![1, 4, 0];
        assert!(!are_indices_valid(&shape, &indices));

        // Test case 3: Mismatched dimensions
        let shape = vec![3, 4, 2];
        let indices = vec![1, 2];
        assert!(!are_indices_valid(&shape, &indices));

        // Test case 4: Empty shape and indices
        let shape = vec![];
        let indices = vec![];
        assert!(are_indices_valid(&shape, &indices));

        // Test case 5: Single dimension with valid index
        let shape = vec![5];
        let indices = vec![3];
        assert!(are_indices_valid(&shape, &indices));

        // Test case 6: Single dimension with invalid index
        let shape = vec![5];
        let indices = vec![5];
        assert!(!are_indices_valid(&shape, &indices));
    }
}
