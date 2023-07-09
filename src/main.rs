#[derive(Debug)]
pub struct MortonArray<T>
where
    T: Copy,
    T: Default,
{
    data: Vec<T>,
    shape: Vec<usize>,
    dimensions: usize,
    magic_numbers: Vec<(u32, usize)>
}

impl<T> MortonArray<T>
where
    T: Copy,
    T: Default,
{
    pub fn new(shape: Vec<usize>) -> Self {
        let dimensions = shape.len();
        let size: usize = shape.iter().product();
        let data = Vec::with_capacity(size);
        let magic_numbers   = generate_magic_numbers(dimensions);
        Self {
            data,
            shape,
            dimensions,
            magic_numbers
        }
    }

    pub fn get(&self, indices: &[usize]) -> Option<&T> {
        let index = self.morton_index(indices)?;
        self.data.get(index)
    }

    pub fn get_mut(&mut self, indices: &[usize]) -> Option<&mut T> {
        let index = self.morton_index(indices)?;
        self.data.get_mut(index)
    }

    pub fn set(&mut self, indices: &[usize], value: T) -> Option<()> {
        let index = self.morton_index(indices)?;
        // println!("indices = {:?}, index = {}", indices, index);
        let len = self.data.len();
        if index >= len {
            self.data.resize_with(index, <T>::default);
            self.data.push(value);
        } else {
            // reassign
            self.data[index] = value;
        }
        Some(())
    }

    fn morton_index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.dimensions {
            return None;
        }
        let mut morton_index = 0;
        let max_index = self
            .shape
            .iter()
            .map(|&dim| dim - 1)
            .collect::<Vec<usize>>();
        for (d, &index) in indices.iter().enumerate() {
            if index > max_index[d] {
                return None;
            }
            morton_index |= self.morton_encode(index as u32) << d;
        }
        // TODO
        Some(morton_index.try_into().unwrap())
    }

    fn morton_encode(&self, x: u32) -> usize {
        let mut word: usize = x as usize;

        for magic_number in self.magic_numbers.iter().rev() {
            word = (word | (word << magic_number.0)) & magic_number.1;
        }
        word
    }

    
}


fn generate_magic_numbers(dimensions: usize) -> Vec<(u32, usize)> {
    let mut num_bits = dimensions as u32;
    let mut left_shift = num_bits - 1;
    let mut result: Vec<(u32, usize)> = Vec::new();

    while num_bits < usize::BITS {
        let mut bitmask: usize = 0;
        for i in 0..usize::BITS {
            if (i + num_bits) % num_bits < (num_bits - left_shift) {
                bitmask += 2_usize.pow(i);
            }
        }

        result.push((left_shift, bitmask));
        left_shift <<= 1;
        num_bits *= 2;
    }

    for i in result.iter() {
        print!("({}, {:X})", i.0, i.1);
    }
    println!("");

    result
}


fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_magic_numbers() {
        assert_eq!(usize::BITS, 64);

        let mut magic_numbers = generate_magic_numbers(2);
        assert_eq!(magic_numbers.len(), 5);
        assert_eq!(magic_numbers[0], (1, 0x5555555555555555));
        assert_eq!(magic_numbers[1], (2, 0x3333333333333333));
        assert_eq!(magic_numbers[2], (4, 0x0F0F0F0F0F0F0F0F));
        assert_eq!(magic_numbers[3], (8, 0x00FF00FF00FF00FF));
        assert_eq!(magic_numbers[4], (16, 0x0000FFFF0000FFFF)); // TODO: useless

        magic_numbers = generate_magic_numbers(3);
        assert_eq!(magic_numbers.len(), 5);
        assert_eq!(magic_numbers[0], (2, 0x9249249249249249));
        assert_eq!(magic_numbers[1], (4, 0x30C30C30C30C30C3)); // TODO: 0x1F00000000FFFF
        assert_eq!(magic_numbers[2], (8, 0xF00F00F00F00F00F)); // TODO: 0x100F00F00F00F00F
        assert_eq!(magic_numbers[3], (16, 0xFF0000FF0000FF)); // TODO: 0x1F0000FF0000FF
        assert_eq!(magic_numbers[4], (32, 0xFFFF00000000FFFF)); // TODO: useless, 0x1F00000000FFFF

        magic_numbers = generate_magic_numbers(4);
        assert_eq!(magic_numbers.len(), 4);
        assert_eq!(magic_numbers[0], (3, 0x1111111111111111));
        assert_eq!(magic_numbers[1], (6, 0x0303030303030303));
        assert_eq!(magic_numbers[2], (12, 0x000F000F000F000F));
        assert_eq!(magic_numbers[3], (24, 0x000000FF000000FF));

        magic_numbers = generate_magic_numbers(5);
        assert_eq!(magic_numbers.len(), 4);
        assert_eq!(magic_numbers[0], (4, 0x1084210842108421));
        assert_eq!(magic_numbers[1], (8, 0x300C0300C0300C03));
        assert_eq!(magic_numbers[2], (16, 0xF0000F0000F0000F));
        assert_eq!(magic_numbers[3], (32, 0x00FF00000000FF));
    }

    #[test]
    fn test_morton_array() {
        let shape = vec![4, 4, 4];
        let mut array: MortonArray<u32> = MortonArray::new(shape.clone());

        // Set and get values
        let mut value = 0;
        let ndim = shape.len();
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
