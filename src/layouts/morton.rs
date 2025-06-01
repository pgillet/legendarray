use crate::{Layout, bit_utils};

struct MortonLayout {
    shape: Vec<usize>,
}

impl MortonLayout {
    fn morton_encode(&self, x: usize, bit_len: usize, num_dim: usize, dim_index: usize) -> usize {
        let n = num_dim - 1;
        let mut z: usize = 0;
        for i in 0usize..bit_len {
            z |= (x & 1 << i) << (n * i + dim_index);
        }
        z
    }
}

impl Layout for MortonLayout {
    fn new(shape: Vec<usize>) -> Self {
        MortonLayout {
            shape,
        }
    }

    fn index(&self, indices: &[usize]) -> Option<usize> {
        let num_dim = self.shape.len();
        if indices.len() != num_dim {
            return None;
        }
        let mut morton_index = 0;
        let max_index = self
            .shape
            .iter()
            .map(|&dim| dim - 1)
            .collect::<Vec<usize>>();

        let bit_len = self
            .shape
            .iter()
            .map(|&dim| bit_utils::log_base_2(dim - 1))
            .collect::<Vec<usize>>();

        for (d, &index) in indices.iter().enumerate() {
            if index > max_index[d] {
                return None;
            }
            morton_index |= self.morton_encode(index, bit_len[d], num_dim, d);
        }
        // TODO
        Some(morton_index.try_into().unwrap())
    }
}

#[cfg(test)]
mod tests {
    use crate::Array;

    use super::*;

    #[test]
    fn test_morton_array() {
        let shape = vec![16, 16, 16];
        let mut array: Array<u32, MortonLayout> = Array::default(shape.clone());

        // Set and get values
        let mut value = 0;
        let ndim: usize = shape.len();
        let mut indices = vec![0; ndim];

        loop {
            // Set value and assert
            value += 1;
            array.set(&indices, value);
            assert_eq!(array.get(&indices), Some(&value));

            // Move to the next indices
            let mut carry = 1;
            for i in 0..ndim {
                indices[i] += carry;
                carry = indices[i] / shape[i];
                indices[i] %= shape[i];

                if carry == 0 {
                    break;
                }
            }

            // Check if all indices have wrapped around
            if carry != 0 {
                break;
            }
        }

        println!("array = {:?}", array.data);

        // Check out-of-bounds access
        assert_eq!(array.get(&[3, 4, 2]), None);

        // Check invalid indices length
        assert_eq!(array.get(&[1, 2]), None);
    }
}
