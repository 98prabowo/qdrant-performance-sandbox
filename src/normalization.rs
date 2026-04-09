#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

pub fn baseline_normalization(vector: Vec<f32>, length: f32) -> Vec<f32> {
    vector.into_iter().map(|x| x / length).collect()
}

/// # Safety
///
/// This function is unsafe because it uses SIMD intrinsics and raw pointer manipulation.
/// The caller must ensure that:
/// 1. The `vector` slice is not empty.
/// 2. The CPU supports the NEON instruction set (aarch64).
pub unsafe fn neon_normalization_optimized(vector: &mut [f32], length: f32) {
    unsafe {
        let n = vector.len();
        if n == 0 {
            return;
        }

        let inv_length = 1.0 / length;
        let inv_vec = vdupq_n_f32(inv_length);
        let ptr = vector.as_mut_ptr();
        let mut i = 0;

        // Phase 1: The "Heavy Lifter" (16 floats at a time)
        // We check if we have at least 16 elements left
        while i + 15 < n {
            let v1 = vld1q_f32(ptr.add(i));
            let v2 = vld1q_f32(ptr.add(i + 4));
            let v3 = vld1q_f32(ptr.add(i + 8));
            let v4 = vld1q_f32(ptr.add(i + 12));

            vst1q_f32(ptr.add(i), vmulq_f32(v1, inv_vec));
            vst1q_f32(ptr.add(i + 4), vmulq_f32(v2, inv_vec));
            vst1q_f32(ptr.add(i + 8), vmulq_f32(v3, inv_vec));
            vst1q_f32(ptr.add(i + 12), vmulq_f32(v4, inv_vec));

            i += 16;
        }

        // Phase 2: The "Gap Filler" (4 floats at a time)
        // If n=18, Phase 1 did 16. Now i=16. 16 + 3 < 18 is false.
        // But we still want to check for groups of 4.
        while i + 3 < n {
            let v = vld1q_f32(ptr.add(i));
            vst1q_f32(ptr.add(i), vmulq_f32(v, inv_vec));
            i += 4;
        }

        // Phase 3: The "Hand Tool" (1 float at a time)
        // Finish whatever is left (0 to 3 elements)
        if i < n {
            for item in &mut vector[i..] {
                *item *= inv_length;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neon_accuracy() {
        let mut vector_neon = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let vector_scalar = vector_neon.clone();
        let length: f32 = 50.0; // Random normalization factor

        unsafe { neon_normalization_optimized(&mut vector_neon, length) };
        let vector_res = baseline_normalization(vector_scalar, length);

        for (neon, scalar) in vector_neon.iter().zip(vector_res.iter()) {
            assert!(
                (neon - scalar).abs() < 1e-6,
                "NEON result {} != Scalar result {}",
                neon,
                scalar
            );
        }
    }

    #[test]
    fn test_neon_alignment() {
        let mut vector = [1.0; 20];
        let length = 2.0;

        let slice = &mut vector[1..18];
        unsafe { neon_normalization_optimized(slice, length) };

        assert_eq!(slice[0], 0.5);
        assert_eq!(slice[16], 0.5);
    }

    #[test]
    fn test_neon_edge_cases() {
        let mut empty: [f32; 0] = [];
        unsafe { neon_normalization_optimized(&mut empty, 1.0) };

        let mut vec = vec![1.0, 2.0];
        unsafe { neon_normalization_optimized(&mut vec, 0.0) };
        assert!(vec[0].is_infinite())
    }

    #[test]
    fn test_neon_tapering() {
        for size in 1..33 {
            let mut vec = vec![1.0; size];
            unsafe { neon_normalization_optimized(&mut vec, 2.0) };
            assert!(vec.iter().all(|&x| x == 0.5), "Failed at size {}", size);
        }
    }
}
