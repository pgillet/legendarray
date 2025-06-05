use crate::{Layout, bit_utils};

pub struct CubicMortonLayout {
    shape: Vec<usize>,
    // num_dim and bit_len_per_dim can be stored if they are frequently used
    // and shape.len() or log_base_2 calculations are non-trivial.
    // For now, we re-calculate them in index() as it's only called there.
}

impl CubicMortonLayout {
    fn morton_encode(&self, x: usize, bit_len: usize, num_dim: usize, dim_index: usize) -> usize {
        let mut z: usize = 0;
        for i in 0usize..bit_len {
            z |= ((x >> i) & 1) << (num_dim * i + dim_index);
        }
        z
    }
}

impl Layout for CubicMortonLayout {
    fn new(shape: Vec<usize>) -> Self {
        if shape.is_empty() {
            // Allow scalar shape, treat as 1-element cubic (1^0 dimensions)
            // Or panic if 0-dim shapes are not desired for this layout
        } else {
            let first_dim_size = shape[0];
            if !bit_utils::is_power_of_2(first_dim_size) {
                panic!("CubicMortonLayout requires all dimension sizes to be powers of 2. Got: {}", first_dim_size);
            }
            for &dim_size in &shape {
                if dim_size != first_dim_size {
                    panic!("CubicMortonLayout requires all dimension sizes to be equal (cubic/square). Got shape: {:?}", shape);
                }
            }
        }
        CubicMortonLayout {
            shape,
        }
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        let num_dim = self.shape.len();
        if indices.len() != num_dim {
            return None;
        }

        // Handle 0-dimensional case (scalar)
        if num_dim == 0 {
            return if indices.is_empty() { Some(0) } else { None };
        }

        let mut morton_index = 0;
        let max_coord_val_for_dim = self.shape[0] - 1; // All dims are same size
        let bit_len_for_dim = bit_utils::log_base_2(max_coord_val_for_dim);

        for (d, &coord_val) in indices.iter().enumerate() {
            if coord_val > max_coord_val_for_dim {
                return None;
            }
            morton_index |= self.morton_encode(coord_val, bit_len_for_dim, num_dim, d);
        }
        Some(morton_index)
    }
}

#[cfg(test)]
mod cubic_morton_tests { // Renamed test module
    use crate::Array; // For test_morton_array_cubic if added
    use super::*;

    #[test]
    fn test_new_cubic_valid() {
        let _ = CubicMortonLayout::new(vec![4, 4]);
        let _ = CubicMortonLayout::new(vec![2, 2, 2]);
        let _ = CubicMortonLayout::new(vec![8]);
        let _ = CubicMortonLayout::new(vec![]); // Scalar
    }

    #[test]
    #[should_panic(expected = "CubicMortonLayout requires all dimension sizes to be equal")]
    fn test_new_cubic_not_square() {
        CubicMortonLayout::new(vec![4, 2]);
    }

    #[test]
    #[should_panic(expected = "CubicMortonLayout requires all dimension sizes to be powers of 2")]
    fn test_new_cubic_not_power_of_2() {
        CubicMortonLayout::new(vec![3, 3]);
    }
    
    #[test]
    #[should_panic(expected = "CubicMortonLayout requires all dimension sizes to be powers of 2")]
    fn test_new_cubic_mixed_not_power_of_2() {
        // This will first fail the "equal" check if dimensions differ,
        // but if they were equal and not PoT, e.g. [6,6]
        CubicMortonLayout::new(vec![6,6]);
    }


    #[test]
    fn test_morton_layout_index_2d_cubic() {
        let shape = vec![4, 4]; // Cubic and power of 2
        let layout = CubicMortonLayout::new(shape.clone());

        // indices are [x,y]. With .enumerate(), x (indices[0]) gets d=0 (dim_index=0).
        // y (indices[1]) gets d=1 (dim_index=1).
        // Morton order: ... y_bit_i x_bit_i ... (e.g., y1x1y0x0 for 2 bits each)
        assert_eq!(layout.index(&[0, 0]), Some(0));  // x=00,y=00 -> x0y0x1y1... -> 0000_b = 0
        assert_eq!(layout.index(&[1, 0]), Some(1));  // x=01,y=00 -> 0001_b = 1
        assert_eq!(layout.index(&[0, 1]), Some(2));  // x=00,y=01 -> 0010_b = 2
        assert_eq!(layout.index(&[1, 1]), Some(3));  // x=01,y=01 -> 0011_b = 3
        assert_eq!(layout.index(&[2, 0]), Some(4));  // x=10,y=00 -> 0100_b = 4
        assert_eq!(layout.index(&[0, 2]), Some(8));  // x=00,y=10 -> 1000_b = 8
        assert_eq!(layout.index(&[2, 2]), Some(12)); // x=10,y=10 -> 1100_b = 12 (x0=0,x1=1; y0=0,y1=1. y1x1y0x0 = 1100)
        assert_eq!(layout.index(&[3, 3]), Some(15)); // x=11,y=11 -> 1111_b = 15

        // Out of bounds
        assert_eq!(layout.index(&[4, 0]), None);
        assert_eq!(layout.index(&[0, 4]), None);

        // Incorrect number of dimensions
        assert_eq!(layout.index(&[1]), None);
        assert_eq!(layout.index(&[1, 1, 1]), None);
    }

    #[test]
    fn test_morton_layout_index_3d_cubic() {
        let shape = vec![2, 2, 2];
        let layout = CubicMortonLayout::new(shape.clone());
        // indices are [x,y,z]. x gets d=0, y gets d=1, z gets d=2.
        // Morton order: ... z_bit_i y_bit_i x_bit_i ... (e.g., z0y0x0 for 1 bit each)
        assert_eq!(layout.index(&[0, 0, 0]), Some(0)); // x0y0z0 = 000_b = 0
        assert_eq!(layout.index(&[1, 0, 0]), Some(1)); // x0y0z0 = 001_b = 1
        assert_eq!(layout.index(&[0, 1, 0]), Some(2)); // x0y0z0 = 010_b = 2
        assert_eq!(layout.index(&[1, 1, 0]), Some(3)); // x0y0z0 = 011_b = 3
        assert_eq!(layout.index(&[0, 0, 1]), Some(4)); // x0y0z0 = 100_b = 4
        assert_eq!(layout.index(&[1, 0, 1]), Some(5)); // x0y0z0 = 101_b = 5
        assert_eq!(layout.index(&[0, 1, 1]), Some(6)); // x0y0z0 = 110_b = 6
        assert_eq!(layout.index(&[1, 1, 1]), Some(7)); // x0y0z0 = 111_b = 7

        assert_eq!(layout.index(&[2,0,0]), None);
    }
    
    #[test]
    fn test_morton_array_cubic() {
        let shape = vec![4, 4];
        let array = Array::<u32, CubicMortonLayout>::arange_sequential(shape.clone());
        // Primarily a visual test or to ensure it doesn't panic with Array struct
        println!("CubicMortonLayout [4,4]:\n{}", array);
        // Expected output based on corrected logic ([1,0] -> 1, [0,1] -> 2, etc.)
        // The data is [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        // Display will use get(), which uses layout.index()
        // [0,0]->0 (val 0) ; [1,0]->1 (val 1) ; [2,0]->4 (val 4) ; [3,0]->5 (val 5)
        // [0,1]->2 (val 2) ; [1,1]->3 (val 3) ; [2,1]->6 (val 6) ; [3,1]->7 (val 7)
        // [0,2]->8 (val 8) ; [1,2]->9 (val 9) ; [2,2]->12(val 12); [3,2]->13(val 13)
        // [0,3]->10(val 10); [1,3]->11(val 11); [2,3]->14(val 14); [3,3]->15(val 15)
        /* Expected visual:
        [[0  1  4  5]
         [2  3  6  7]
         [8  9 12 13]
         [10 11 14 15]]
        */
         assert_eq!(array.get(&[0,0]), Some(&0));
         assert_eq!(array.get(&[1,0]), Some(&1));
         assert_eq!(array.get(&[0,1]), Some(&2));
         assert_eq!(array.get(&[1,1]), Some(&3));
         assert_eq!(array.get(&[2,0]), Some(&4));
         assert_eq!(array.get(&[3,3]), Some(&15));


        let shape_3d = vec![2,2,2];
        let array_3d = Array::<u32, CubicMortonLayout>::arange_sequential(shape_3d.clone());
        println!("CubicMortonLayout [2,2,2]:\n{}", array_3d);
        assert_eq!(array_3d.get(&[1,1,1]), Some(&7));
    }
} 