use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
mod neon_bench {
    use super::*;
    use qdrant_performance_sandbox::norm_neon::{
        baseline_norm_neon_alloc, baseline_norm_neon_in_place, optimized_norm_neon_simd,
    };

    pub fn run(c: &mut Criterion) {
        let mut group = c.benchmark_group("Normalization/Neon");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        for dims in [384, 768, 1536, 4096, 16384, 65536, 262144, 1048576].iter() {
            let vector = vec![1.0f32; *dims];
            let length = 10.0;

            group.bench_with_input(BenchmarkId::new("Baseline", dims), dims, |b, _| {
                b.iter(|| black_box(baseline_norm_neon_alloc(vector.clone(), length)))
            });

            group.bench_with_input(BenchmarkId::new("In-Place", dims), dims, |b, _| {
                b.iter_batched(
                    || vector.clone(),
                    |mut data| {
                        baseline_norm_neon_in_place(&mut data, length);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            });

            group.bench_with_input(BenchmarkId::new("Proposed", dims), dims, |b, _| {
                b.iter_batched(
                    || vector.clone(),
                    |mut data| unsafe {
                        optimized_norm_neon_simd(&mut data, length);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        group.finish();
    }
}

#[cfg(target_arch = "x86_64")]
mod avx_bench {
    use super::*;
    use qdrant_performance_sandbox::norm_avx::{
        baseline_norm_avx_alloc, baseline_norm_avx_in_place, optimized_norm_avx_simd,
    };

    pub fn run(c: &mut Criterion) {
        let mut group = c.benchmark_group("Normalization/AVX");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        for dims in [384, 768, 1536, 4096, 16384, 65536, 262144, 1048576].iter() {
            let vector = vec![1.0f32; *dims];
            let length = 10.0;

            group.bench_with_input(BenchmarkId::new("Baseline", dims), dims, |b, _| {
                b.iter(|| black_box(baseline_norm_avx_alloc(vector.clone(), length)))
            });

            group.bench_with_input(BenchmarkId::new("In-Place", dims), dims, |b, _| {
                b.iter_batched(
                    || vector.clone(),
                    |mut data| {
                        baseline_norm_avx_in_place(&mut data, length);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            });

            group.bench_with_input(BenchmarkId::new("Proposed", dims), dims, |b, _| {
                b.iter_batched(
                    || vector.clone(),
                    |mut data| unsafe {
                        optimized_norm_avx_simd(&mut data, length);
                        black_box(data)
                    },
                    BatchSize::LargeInput,
                );
            });
        }

        group.finish();
    }
}

fn run_benchmarks(c: &mut Criterion) {
    #[cfg(target_arch = "aarch64")]
    neon_bench::run(c);

    #[cfg(target_arch = "x86_64")]
    avx_bench::run(c);
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
