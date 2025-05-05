#![feature(test)]

extern crate test;
use test::Bencher;

use legendarray::{Array, Layout};
use legendarray::row_major_order::base::RowMajorOrderLayout;
use legendarray::row_major_order::par::RowMajorOrderLayoutPar;

const SIZE: usize = 250;

#[bench]
fn bench_row_major_without_precomputed_strides(bench: &mut Bencher) {
    let shape = vec![SIZE, SIZE, SIZE];
    let layout = RowMajorOrderLayout::new(shape.clone());
    let mut array = Array::<i32, RowMajorOrderLayout>::new(shape);

    bench.iter(|| {
        for i in 0..SIZE {
            for j in 0..SIZE {
                for k in 0..SIZE {
                    array.set(&[i, j, k], i as i32 + j as i32 + k as i32);
                }
            }
        }
    });
}

#[bench]
fn bench_row_major_with_precomputed_strides(bench: &mut Bencher) {
    let shape = vec![SIZE, SIZE, SIZE];
    let layout = RowMajorOrderLayoutPar::new(shape.clone());
    let mut array = Array::<i32, RowMajorOrderLayoutPar>::new(shape);

    bench.iter(|| {
        for i in 0..SIZE {
            for j in 0..SIZE {
                for k in 0..SIZE {
                    array.set(&[i, j, k], i as i32 + j as i32 + k as i32);
                }
            }
        }
    });
}
