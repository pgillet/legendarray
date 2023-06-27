#[derive(Debug)]
pub struct MortonArray<T>
where
    T: Copy,
    T: Default,
{
    data: Vec<T>,
    shape: Vec<usize>,
    dimensions: usize,
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
        Self {
            data,
            shape,
            dimensions,
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
        println!("indices = {:?}, index = {}", indices, index);
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
            morton_index |= morton_decode_32(index as u32) << d;
        }
        // TODO
        Some(morton_index.try_into().unwrap())
    }
}

fn morton_decode_32(x: u32) -> usize {
    let mut word: usize = x as usize;
    //word &= 0x1FFFFF; // we only look at the first 21 bits
    word = (word | word << 32) & 0x1F00000000FFFF; // shift left 32 bits, OR with self, and 00011111000000000000000000000000000000001111111111111111
    word = (word | word << 16) & 0x1F0000FF0000FF; // shift left 32 bits, OR with self, and 00011111000000000000000011111111000000000000000011111111
    word = (word | word << 8) & 0x100F00F00F00F00F; // shift left 32 bits, OR with self, and 0001000000001111000000001111000000001111000000001111000000000000
    word = (word | word << 4) & 0x10C30C30C30C30C3; // shift left 32 bits, OR with self, and 0001000011000011000011000011000011000011000011000011000100000000
    word = (word | word << 2) & 0x1249249249249249;

    //word = (word | (word << 16)) & 0x0000FFFF0000FFFF;
    //word = (word | (word << 8)) & 0x00FF00FF00FF00FF;
    //word = (word | (word << 4)) & 0x0F0F0F0F0F0F0F0F;
    //word = (word | (word << 2)) & 0x3333333333333333;
    //word = (word | (word << 1)) & 0x5555555555555555;
    return word;
}

fn morton_decode_16(x: u16) -> usize {
    let mut word: usize = x as usize;
    //word &= 0x3ff;
    word = (word | (word << 16)) & 0x30000FF;
    word = (word | (word << 8)) & 0x300F00F;
    word = (word | (word << 4)) & 0x30C30C3;
    word = (word | (word << 2)) & 0x9249249;
    return word;
}

fn main() {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morton_array() {
        let shape = vec![4, 4, 4];
        let mut array: MortonArray<u32> = MortonArray::new(shape.clone());

        // Set and get values
        let mut value = 0;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    value += 1;
                    array.set(&[i, j, k], value);
                    println!("array = {:?}", array.data);
                    assert_eq!(array.get(&[i, j, k]), Some(&value));
                }
            }
        }

        println!("array = {:?}", array.data);

        // Check out-of-bounds access
        assert_eq!(array.get(&[3, 4, 2]), None);

        // Check invalid indices length
        assert_eq!(array.get(&[1, 2]), None);
    }
}
