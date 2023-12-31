//! # zorder
//!
//! This crate provides functions to convert between 2D coordinates and a Z-order curve index.
//! The Z-order curve is a space-filling curve that maps 2D coordinates to a 1D index.
//! The Z-order curve is also known as Morton code. Advantage of the Z-order curve is that
//! is may be used to efficiently store 2D data in a 1D array as it preserves locality, and
//! is therefore cache-friendly.
//!
//! This crate provides two implementations of the Z-order curve, one using a software
//! implementation supported by all platforms and one using bmi2 instructions
//! supported by modern x86_64 CPUs.
//!
//!
//! # Examples
//!
//! Basic usage with software implementation:
//!
//! ```
//! use zorder::{index_of, coord_of};
//!
//! let idx = index_of((1, 1));
//! assert_eq!(idx, 3);
//!
//! let coord = coord_of(idx);
//! assert_eq!(coord, (1, 1));
//! ```
//!
//! Basic usage with bmi2 implementation:
//!
//! ```
//! #[cfg(target_arch = "x86_64")]
//! {
//!     use zorder::bmi2::{index_of, coord_of};
//!
//!     if is_x86_feature_detected!("bmi2") {
//!         let idx = unsafe { index_of((1, 1)) };
//!         assert_eq!(idx, 3);
//!
//!         let coord = unsafe { coord_of(idx) };
//!         assert_eq!(coord, (1, 1));
//!     }
//! }
//! ```

#![no_std]

/// Returns the Z-order curve index of the given 2D coordinates.
///
/// # Examples
///
/// ```
/// use zorder::index_of;
///
/// let idx = index_of((1, 1));
/// assert_eq!(idx, 3);
/// ```
#[inline]
pub fn index_of((x, y): (u16, u16)) -> u32 {
    // Adapted originally from:
    // http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN
    //
    // This implementation uses u64 instead of u32 to interleave both x and y
    // in parallel in single pass.
    let packed = (x as u64) | ((y as u64) << 32);

    let first = (packed | (packed << 8)) & 0x00FF00FF00FF00FF;
    let second = (first | (first << 4)) & 0x0F0F0F0F0F0F0F0F;
    let third = (second | (second << 2)) & 0x3333333333333333;
    let fourth = (third | (third << 1)) & 0x5555555555555555;

    let x = fourth;
    let y = fourth >> 31;
    (x | y) as u32
}

/// Returns the 2D coordinates of the given Z-order curve index.
///
/// # Examples
///
/// ```
/// use zorder::coord_of;
///
/// let coord = coord_of(3);
/// assert_eq!(coord, (1, 1));
/// ```
#[inline]
pub fn coord_of(idx: u32) -> (u16, u16) {
    // Adapted originally from:
    // https://stackoverflow.com/questions/4909263/how-to-efficiently-de-interleave-bits-inverse-morton
    //
    // Similar to the `index_of` function, this implementation uses u64 to
    // deinterleave both x and y in parallel in single pass.
    let wide_idx = idx as u64;
    let packed = (wide_idx & 0x55555555) | ((wide_idx & 0xAAAAAAAA) << 31);

    let first = (packed | (packed >> 1)) & 0x3333333333333333;
    let second = (first | (first >> 2)) & 0x0F0F0F0F0F0F0F0F;
    let third = (second | (second >> 4)) & 0x00FF00FF00FF00FF;
    let fourth = third | (third >> 8);

    let x = fourth as u16;
    let y = (fourth >> 32) as u16;
    (x, y)
}

#[cfg(target_arch = "x86_64")]
pub mod bmi2 {
    /// Returns the Z-order curve index of the given 2D coordinates.
    ///
    /// This function requires the bmi2 instruction set, but is much faster
    /// than the software implementation.
    ///
    /// # Safety
    ///
    /// This function is safe to call only if the `bmi2` x86_64 feature is
    /// supported by the CPU. This can be checked at runtime:
    ///
    /// ```
    /// #[cfg(target_arch = "x86_64")]
    /// {
    ///     if is_x86_feature_detected!("bmi2") {
    ///         // ...
    ///     }
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use zorder::bmi2::index_of;
    ///
    /// #[cfg(target_arch = "x86_64")]
    /// {
    ///     if is_x86_feature_detected!("bmi2") {
    ///         let idx = unsafe { index_of((1, 1)) };
    ///         assert_eq!(idx, 3);
    ///     }
    /// }
    /// ```
    #[inline]
    #[target_feature(enable = "bmi2")]
    pub unsafe fn index_of((x, y): (u16, u16)) -> u32 {
        use core::arch::x86_64::_pdep_u32;

        let x = _pdep_u32(x as u32, 0x55555555);
        let y = _pdep_u32(y as u32, 0xAAAAAAAA);
        x | y
    }

    /// Returns the 2D coordinates of the given Z-order curve index.
    ///
    /// This function requires the bmi2 instruction set, but is much faster
    /// than the software implementation.
    ///
    /// # Safety
    ///
    /// This function is safe to call only if the `bmi2` x86_64 feature is
    /// supported by the CPU. This can be checked at runtime:
    ///
    /// ```
    /// #[cfg(target_arch = "x86_64")]
    /// {
    ///     if is_x86_feature_detected!("bmi2") {
    ///         // ...
    ///     }
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use zorder::bmi2::coord_of;
    ///
    /// #[cfg(target_arch = "x86_64")]
    /// {
    ///     if is_x86_feature_detected!("bmi2") {
    ///         let coord = unsafe { coord_of(3) };
    ///         assert_eq!(coord, (1, 1));
    ///     }
    /// }
    /// ```
    #[inline]
    #[target_feature(enable = "bmi2")]
    pub unsafe fn coord_of(idx: u32) -> (u16, u16) {
        use core::arch::x86_64::_pext_u32;

        let x = _pext_u32(idx, 0x55555555);
        let y = _pext_u32(idx, 0xAAAAAAAA);
        (x as u16, y as u16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_of_and_coord_of() {
        for i in 0..1000 {
            let xy = coord_of(i);
            println!("{:?} = {}", xy, i);
            assert_eq!(index_of(xy), i);
        }
    }
}