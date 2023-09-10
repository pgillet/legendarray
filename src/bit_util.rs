/// Find the log base 2 of an integer with the MSB N set
/// in O(N) operations (the obvious way)
fn log_base_2(i: usize) -> u32 {
    let mut v: usize = i;
    let mut r: u32 = 1;
    loop {
        v >>= 1;

        if v == 0 {
            break;
        }

        r += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_base_2() {
        assert_eq!(log_base_2(5), 3);
        assert_eq!(log_base_2(16), 5);
        assert_eq!(log_base_2(77), 7);
        assert_eq!(log_base_2(128), 8);
    }
}
