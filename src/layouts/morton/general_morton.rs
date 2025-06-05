use crate::{Layout, bit_utils};

pub struct GeneralMortonLayout {
    shape: Vec<usize>,
    // Pre-calculating bit_len and max_index vectors in new() could be an optimization.
    // For now, they are calculated on each call to index().
}

impl GeneralMortonLayout {
    fn morton_encode(&self, x: usize, bit_len: usize, num_dim: usize, dim_index: usize) -> usize {
        let mut z: usize = 0;
        for i in 0usize..bit_len {
            z |= ((x >> i) & 1) << (num_dim * i + dim_index);
        }
        z
    }
}

impl Layout for GeneralMortonLayout {
    fn new(shape: Vec<usize>) -> Self {
        if !shape.is_empty() {
            for &dim_size in &shape {
                if !bit_utils::is_power_of_2(dim_size) {
                    panic!("GeneralMortonLayout requires all dimension sizes to be powers of 2. Got shape: {:?}, offending dim: {}", shape, dim_size);
                }
            }
        }
        GeneralMortonLayout {
            shape,
        }
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        let num_dim = self.shape.len();
        if indices.len() != num_dim {
            return None;
        }

        if num_dim == 0 {
            return if indices.is_empty() { Some(0) } else { None };
        }

        let mut morton_index = 0;
        // These need to be vectors because dimensions can have different sizes.
        let max_coords_per_dim = self
            .shape
            .iter()
            .map(|&dim_size| dim_size - 1)
            .collect::<Vec<usize>>();

        let bit_lengths_per_dim = self
            .shape
            .iter()
            .map(|&dim_size| bit_utils::log_base_2(dim_size - 1))
            .collect::<Vec<usize>>();

        for (d, &coord_val) in indices.iter().enumerate() {
            if coord_val > max_coords_per_dim[d] {
                return None;
            }
            morton_index |= self.morton_encode(coord_val, bit_lengths_per_dim[d], num_dim, d);
        }
        Some(morton_index)
    }
}

#[cfg(test)]
mod general_morton_tests { // Renamed test module
    use crate::Array;
    use super::*;

    #[test]
    fn test_new_general_valid() {
        let _ = GeneralMortonLayout::new(vec![4, 4]);
        let _ = GeneralMortonLayout::new(vec![2, 2, 2]);
        let _ = GeneralMortonLayout::new(vec![4, 2]); // Mixed, but power of 2
        let _ = GeneralMortonLayout::new(vec![8, 1, 4]); // Mixed, with 1 (2^0)
        let _ = GeneralMortonLayout::new(vec![]);
    }

    #[test]
    #[should_panic(expected = "GeneralMortonLayout requires all dimension sizes to be powers of 2")]
    fn test_new_general_not_power_of_2_square() {
        GeneralMortonLayout::new(vec![3, 3]);
    }

    #[test]
    #[should_panic(expected = "GeneralMortonLayout requires all dimension sizes to be powers of 2")]
    fn test_new_general_not_power_of_2_mixed() {
        GeneralMortonLayout::new(vec![4, 6]);
    }

    #[test]
    fn test_morton_layout_index_2d_general_square() {
        let shape = vec![4, 4];
        let layout = GeneralMortonLayout::new(shape.clone());
        assert_eq!(layout.index(&[0, 0]), Some(0));
        assert_eq!(layout.index(&[1, 0]), Some(1));
        assert_eq!(layout.index(&[0, 1]), Some(2));
        assert_eq!(layout.index(&[1, 1]), Some(3));
        assert_eq!(layout.index(&[2, 0]), Some(4));
        assert_eq!(layout.index(&[0, 2]), Some(8));
        assert_eq!(layout.index(&[2, 2]), Some(12));
        assert_eq!(layout.index(&[3, 3]), Some(15));
    }

    #[test]
    fn test_morton_layout_index_3d_general_cubic() {
        let shape = vec![2, 2, 2];
        let layout = GeneralMortonLayout::new(shape.clone());
        assert_eq!(layout.index(&[0,0,0]), Some(0));
        assert_eq!(layout.index(&[1,0,0]), Some(1));
        assert_eq!(layout.index(&[0,1,0]), Some(2));
        assert_eq!(layout.index(&[1,1,1]), Some(7));
    }

    #[test]
    fn test_morton_layout_index_mixed_shape_general() {
        // Shape [4,2] => x (dim0): 0..3 (2 bits), y (dim1): 0..1 (1 bit)
        // x gets d=0, y gets d=1.
        // num_dim = 2.
        // Morton order: y_bit_i x_bit_i (y0x0 for bit 0 of each, y0x1 for y0 and x1 etc)
        // Example: [x,y] = [1,0]. x=1 (01), y=0 (0)
        // x-contrib (d=0): encode(1, 2_bits, 2_dims, 0_dim_idx) -> ((1>>0)&1)<<(2*0+0) | ((0>>1)&1)<<(2*1+0) = 1
        // y-contrib (d=1): encode(0, 1_bit,  2_dims, 1_dim_idx) -> ((0>>0)&1)<<(2*0+1) = 0
        // Result: 1 | 0 = 1.
        let shape = vec![4, 2];
        let layout = GeneralMortonLayout::new(shape.clone());
        assert_eq!(layout.index(&[0, 0]), Some(0));  // x(00) y(0) -> x0=0,x1=0; y0=0. (y0x1x0 pattern) -> 000 = 0
        assert_eq!(layout.index(&[1, 0]), Some(1));  // x(01) y(0) -> x0=1,x1=0; y0=0. (y0x1x0 pattern) -> 001 = 1
        assert_eq!(layout.index(&[2, 0]), Some(4));  // x(10) y(0) -> x0=0,x1=1; y0=0. (y0x1x0 pattern) -> 100 = 4
        assert_eq!(layout.index(&[3, 0]), Some(5));  // x(11) y(0) -> x0=1,x1=1; y0=0. (y0x1x0 pattern) -> 101 = 5

        assert_eq!(layout.index(&[0, 1]), Some(2));  // x(00) y(1) -> x0=0,x1=0; y0=1. (y0x1x0 pattern) -> 010 = 2
        assert_eq!(layout.index(&[1, 1]), Some(3));  // x(01) y(1) -> x0=1,x1=0; y0=1. (y0x1x0 pattern) -> 011 = 3
        assert_eq!(layout.index(&[2, 1]), Some(6));  // x(10) y(1) -> x0=0,x1=1; y0=1. (y0x1x0 pattern) -> 110 = 6
        assert_eq!(layout.index(&[3, 1]), Some(7));  // x(11) y(1) -> x0=1,x1=1; y0=1. (y0x1x0 pattern) -> 111 = 7

        // Out of bounds for shape, but valid coordinates for a conceptual bounding box
        // These tests confirm the Morton calculation itself.
        // For shape [8,2] (x: 3 bits, y: 1 bit)
        // Let layout_8_2 = GeneralMortonLayout::new(vec![8,2]);
        // assert_eq!(layout_8_2.index(&[7,1]), Some( ... )); // (y0 x2 x1 x0) ... y0=1, x0=1,x1=1,x2=1 -> 1111 = 15

        // Check out-of-bounds for defined shape
        assert_eq!(layout.index(&[4,0]), None);
        assert_eq!(layout.index(&[0,2]), None);
    }

    #[test]
    fn test_morton_array_general_mixed_shape() {
        let shape = vec![4, 2]; // Total 8 elements (0-7)
        let array = Array::<u32, GeneralMortonLayout>::arange_sequential(shape.clone());
        println!("GeneralMortonLayout [4,2]:\n{}", array);
        // Data: [0,1,2,3,4,5,6,7]
        // Coords -> Morton Index (Value from data if in bounds)
        // [0,0] -> 0 (val 0)
        // [1,0] -> 1 (val 1)
        // [2,0] -> 4 (val 4)
        // [3,0] -> 5 (val 5)
        // [0,1] -> 2 (val 2)
        // [1,1] -> 3 (val 3)
        // [2,1] -> 6 (val 6)
        // [3,1] -> 7 (val 7)
        /* Expected output:
        [[0 1]
         [4 5]
         [2 3]
         [6 7]] <-- This is if printed by x, then y. But display is by shape.
        Correct expected Array Display for [4,2] shape (x varies fastest in inner loop of display):
        [[0 1]
         [2 3]
         [4 5]
         [6 7]] <-- My Array Display fn prints this way.
        Let's verify using .get()
        */
        assert_eq!(array.get(&[0,0]), Some(&0));
        assert_eq!(array.get(&[1,0]), Some(&1));
        assert_eq!(array.get(&[0,1]), Some(&2));
        assert_eq!(array.get(&[1,1]), Some(&3));
        assert_eq!(array.get(&[2,0]), Some(&4));
        assert_eq!(array.get(&[3,0]), Some(&5));
        assert_eq!(array.get(&[2,1]), Some(&6));
        assert_eq!(array.get(&[3,1]), Some(&7));

        // Example from user output where zeros appeared
        // This shape will cause Morton indices that might be out of direct product range if not careful
        // but here all resulting Morton indices (0-7 for [4,2]) are < 8, so no zeros from out-of-bounds.
        // If shape was e.g. [6,2], dim0 is 3 bits, dim1 is 1 bit.
        // Indices like [5,1] -> x=5 (101), y=1 (1)
        // x-contrib (d=0): encode(5, 3, 2, 0) -> (1<<0)|(0<<2)|(1<<4) = 1|0|16 = 17
        // y-contrib (d=1): encode(1, 1, 2, 1) -> (1<<1) = 2
        // Morton = 17 | 2 = 19. Product is 6*2=12. Index 19 is out of bounds.
        // This is the expected behavior for standard Morton on non-aligned mixed shapes.
        let shape_6_2 = vec![6,2]; // Not all power of 2, so should panic at new()
                                   // Ah, user said "Only the case where one dimension is a multiple of the other."
                                   // And "Check also in both implementations that the given dimensions are power of 2"
                                   // So [6,2] should be rejected by new(). My panic tests cover this.
    }
    
    #[test]
    fn test_morton_array_general_scalar() {
        let shape = vec![];
        let array = Array::<u32, GeneralMortonLayout>::arange_sequential(shape.clone());
        println!("GeneralMortonLayout SCALAR:\n{}", array);
        assert_eq!(array.get(&[]), Some(&0));
    }
} 