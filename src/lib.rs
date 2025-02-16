mod morton;
mod column_major_order;
mod row_major_order;
mod bit_util;


pub trait Layout {
    fn new(shape: Vec<usize>) -> Self;
    fn index(&self, indices: &[usize]) -> Option<usize>;
}

#[derive(Debug)]
pub struct Array<T, L: Layout>
where
    T: Copy,
    T: Default,
{
    data: Vec<T>,
    shape: Vec<usize>,
    dimensions: usize,
    layout: L,
}

impl<T, L: Layout> Array<T, L>
where
    T: Copy,
    T: Default,
{
    pub fn new(shape: Vec<usize>) -> Self {
        let dimensions = shape.len();
        let size: usize = shape.iter().product();
        let data = Vec::with_capacity(size);
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
}
