#![feature(test)]

extern crate test;
use test::Bencher;

use legendarray::{Array, Layout};
use legendarray::layouts::RowMajorOrderLayout;

const SIZE: usize = 250;

#[bench]
fn bench_row_major(bench: &mut Bencher) {
    let shape = vec![SIZE, SIZE, SIZE];
    let layout = RowMajorOrderLayout::new(shape.clone());
    let mut array = Array::<i32, RowMajorOrderLayout>::default(shape);

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
