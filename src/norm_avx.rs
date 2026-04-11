#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_feature = "avx")]
#[inline(always)]
pub fn baseline_norm_avx(vector: Vec<f32>, length: f32) -> Vec<f32> {
    vector.into_iter().map(|x| x / length).collect()
}

#[cfg(target_feature = "avx")]
#[allow(clippy::missing_safety_doc)]
#[inline(always)]
pub unsafe fn optimized_norm_avx(vector: &mut [f32], length: f32) {
    unsafe {
        let n = vector.len();
        if n == 0 {
            return;
        }

        let inv_length = 1.0 / length;
        let inv_vec = _mm256_set1_ps(inv_length);
        let mut_ptr = vector.as_mut_ptr();

        let mut i = 0;

        // Phase 1: The "Heavy Lifter" (32 floats at a time)
        // We check if we have at least 32 elements left
        while i + 15 < n {
            let v1 = _mm256_loadu_ps(mut_ptr.add(i));
            let v2 = _mm256_loadu_ps(mut_ptr.add(i + 8));

            _mm256_storeu_ps(mut_ptr.add(i), _mm256_mul_ps(v1, inv_vec));
            _mm256_storeu_ps(mut_ptr.add(i + 8), _mm256_mul_ps(v2, inv_vec));

            i += 16;
        }

        // Phase 2: The "Gap Filler" (16 floats at a time)
        // If n=34, Phase 1 did 32. Now i=32. 32 + 7 < 34 is false.
        // But we still want to check for groups of 8.
        while i + 7 < n {
            let v = _mm256_loadu_ps(mut_ptr.add(i));
            _mm256_storeu_ps(mut_ptr.add(i), _mm256_mul_ps(v, inv_vec));
            i += 8;
        }

        // Phase 3: The "Hand Tool" (1 float at a time)
        // Finish whatever is left (0 to 3 elements)
        for item in vector.iter_mut().take(n).skip(i) {
            *item *= inv_length;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avx_accuracy() {
        let mut vector_avx = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0,
        ];
        let vector_scalar = vector_avx.clone();
        let length: f32 = 50.0; // Random normalization factor

        unsafe { optimized_norm_avx(&mut vector_avx, length) };
        let vector_res = baseline_norm_avx(vector_scalar, length);

        for (neon, scalar) in vector_avx.iter().zip(vector_res.iter()) {
            let tol = 1e-6_f32.max(8.0 * f32::EPSILON * neon.abs().max(scalar.abs()).max(1.0));
            assert!(
                (neon - scalar).abs() <= tol,
                "NEON result {} != Scalar result {}",
                neon,
                scalar
            );
        }
    }

    #[test]
    fn test_avx_alignment() {
        let mut vector = [1.0; 20];
        let length = 2.0;

        let slice = &mut vector[1..18];
        unsafe { optimized_norm_avx(slice, length) };

        assert_eq!(slice[0], 0.5);
        assert_eq!(slice[16], 0.5);
    }

    #[test]
    fn test_avx_edge_cases() {
        let mut empty: [f32; 0] = [];
        unsafe { optimized_norm_avx(&mut empty, 1.0) };

        let mut vec = vec![1.0, 2.0];
        unsafe { optimized_norm_avx(&mut vec, 0.0) };
        assert!(vec[0].is_infinite())
    }

    #[test]
    fn test_avx_tapering() {
        for size in 1..33 {
            let mut vec = vec![1.0; size];
            unsafe { optimized_norm_avx(&mut vec, 2.0) };
            assert!(vec.iter().all(|&x| x == 0.5), "Failed at size {}", size);
        }
    }
}
