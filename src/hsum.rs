#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// # Safety
///
/// This function is unsafe because it uses AArch64 NEON intrinsics.
/// The caller must ensure that the target architecture is aarch64 with NEON support.
pub unsafe fn baseline_sum(
    sum1: float32x4_t,
    sum2: float32x4_t,
    sum3: float32x4_t,
    sum4: float32x4_t,
) -> f32 {
    unsafe { vaddvq_f32(sum1) + vaddvq_f32(sum2) + vaddvq_f32(sum3) + vaddvq_f32(sum4) }
}

/// # Safety
///
/// This function is unsafe because it uses AArch64 NEON intrinsics.
/// The caller must ensure that the target architecture is aarch64 and that NEON
/// instructions are available.
pub unsafe fn optimized_sum(
    sum1: float32x4_t,
    sum2: float32x4_t,
    sum3: float32x4_t,
    sum4: float32x4_t,
) -> f32 {
    unsafe { vaddvq_f32(vaddq_f32(vaddq_f32(sum1, sum2), vaddq_f32(sum3, sum4))) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hsum() {
        unsafe {
            let sum1 = vdupq_n_f32(1.1);
            let sum2 = vdupq_n_f32(2.2);
            let sum3 = vdupq_n_f32(3.3);
            let sum4 = vdupq_n_f32(4.4);

            let baseline = baseline_sum(sum1, sum2, sum3, sum4);
            let optimized = optimized_sum(sum1, sum2, sum3, sum4);

            assert_eq!(baseline, optimized);
        }
    }
}
