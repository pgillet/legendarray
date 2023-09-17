/// Find the log base 2 of an integer with the MSB N set
/// in O(N) operations (the obvious way)
/// See http://www.graphics.stanford.edu/~seander/bithacks.html#IntegerLogObvious
pub fn log_base_2(i: usize) -> usize {
    let mut v: usize = i;
    let mut r: usize = 1;
    loop {
        v >>= 1;

        if v == 0 {
            break;
        }

        r += 1;
    }
    r
}

/// Determine if an integer is a power of 2
/// See http://www.graphics.stanford.edu/~seander/bithacks.html#DetermineIfPowerOf2
pub fn is_power_of_2(v: usize) -> bool {
    v > 1 && (v & (v - 1)) == 0
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

    #[test]
    fn test_is_power_of_2() {
        assert_eq!(is_power_of_2(0), false);
        assert_eq!(is_power_of_2(1), false);
        assert_eq!(is_power_of_2(2), true);
        assert_eq!(is_power_of_2(3), false);
        assert_eq!(is_power_of_2(4), true);
        assert_eq!(is_power_of_2(5), false);
        assert_eq!(is_power_of_2(6), false);
        assert_eq!(is_power_of_2(7), false);
        assert_eq!(is_power_of_2(8), true);
        assert_eq!(is_power_of_2(16), true);
        assert_eq!(is_power_of_2(32), true);
        assert_eq!(is_power_of_2(64), true);
        assert_eq!(is_power_of_2(128), true);
        assert_eq!(is_power_of_2(256), true);
    }

}
