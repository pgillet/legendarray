use crate::Layout;

pub struct RowMajorOrderLayoutPar {
    shape: Vec<usize>,
    strides: Vec<usize>, // Pre-computed strides for faster index calculation
}

impl Layout for RowMajorOrderLayoutPar {
    fn new(shape: Vec<usize>) -> Self {
        // Compute strides at initialization time
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        RowMajorOrderLayoutPar {
            shape,
            strides,
        }
    }

    // Calculate index for a single set of indices
    fn index(&self, indices: &[usize]) -> Option<usize> {
        if indices.len() != self.shape.len() {
            return None;
        }

        // Calculate the linear index using pre-computed strides
        let mut linear_index = 0;
        for (dim, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[dim] {
                return None;
            }
            linear_index += idx * self.strides[dim];
        }

        Some(linear_index)

        // for (i, &idx) in indices.iter().enumerate() {
        //     if idx >= self.shape[i] {
        //         return None;
        //     }
        // }
        //
        // // Calculate the linear index using pre-computed strides
        // Some(indices.iter().enumerate()
        //     .map(|(dim, &idx)| idx * self.strides[dim])
        //     .sum())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use super::*;
    use crate::{Array, Layout};

    #[test]
    fn test_index_1d() {
        let shape = vec![5];
        let indices = vec![2];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(2));
    }

    #[test]
    fn test_index_2d() {
        let shape = vec![3, 4];
        let indices = vec![1, 2];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(6));
    }

    #[test]
    fn test_index_2d_bis() {
        let shape = vec![3, 4];
        let indices = vec![2, 1];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(9));
    }

    #[test]
    fn test_index_2d_ter() {
        let shape = vec![3, 3];
        let indices = vec![2, 2];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(8));
    }

    #[test]
    fn test_index_3d() {
        let shape = vec![3, 4, 5];
        let indices = vec![2, 3, 4];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(59));
    }

    #[test]
    fn test_index_3d_bis() {
        let shape = vec![3, 3, 3];
        let indices = vec![2, 1, 1];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(22));
    }

    #[test]
    fn test_index_3d_ter() {
        let shape = vec![3, 4, 5];
        let indices = vec![1, 2, 2];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(32));
    }

    #[test]
    fn test_index_4d() {
        let shape = vec![2, 3, 4, 5];
        let indices = vec![1, 2, 3, 4];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), Some(119));
    }

    #[test]
    fn test_index_mismatch_dimensions() {
        let shape = vec![3, 4];
        let indices = vec![1, 2, 3];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), None);
    }

    #[test]
    fn test_index_index_out_of_bounds() {
        let shape = vec![3, 4];
        let indices = vec![3, 4];
        let layout = RowMajorOrderLayoutPar::new(shape);
        assert_eq!(layout.index(&indices), None);
    }
}