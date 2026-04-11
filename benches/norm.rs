use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
mod neon_bench {
    use super::*;
    use criterion::BatchSize;
    use qdrant_performance_sandbox::norm_neon::{baseline_norm_neon, optimized_norm_neon};

    const SMALL_DIMS: usize = 1532; // Standard AI vector size (e.g., OpenAI embeddings)
    const BIG_DIMS: usize = 1_000_000;

    pub fn run(c: &mut Criterion) {
        let small_vector_base = vec![0.5f32; SMALL_DIMS];
        let big_vector_base = vec![0.5f32; BIG_DIMS];
        let length = 10.0f32; // Pre-calculated length for the benchmark

        let mut group = c.benchmark_group("Normalization/Neon");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        group.bench_function("Small/Baseline", |b| {
            b.iter(|| {
                let input = black_box(small_vector_base.clone());
                baseline_norm_neon(input, length);
            });
        });

        group.bench_function("Small/Proposed", |b| {
            b.iter(|| {
                let mut input = black_box(small_vector_base.clone());
                unsafe {
                    optimized_norm_neon(&mut input, length);
                }
            });
        });

        group.bench_function("Big/Baseline", |b| {
            b.iter_batched(
                || big_vector_base.clone(),
                |input| baseline_norm_neon(input, length),
                BatchSize::SmallInput,
            );
        });

        group.bench_function("Big/Proposed", |b| {
            b.iter_batched(
                || big_vector_base.clone(),
                |mut input| unsafe { optimized_norm_neon(&mut input, length) },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

#[cfg(target_arch = "x86_64")]
mod avx_bench {
    use super::*;
    use criterion::BenchmarkId;
    use qdrant_performance_sandbox::norm_avx::{baseline_norm_avx, optimized_norm_avx};

    pub fn run(c: &mut Criterion) {
        let mut group = c.benchmark_group("Normalization Sweep");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        // Test from 1024 dims up to 1M dims
        for dims in [1024, 4096, 16384, 65536, 262144, 1048576].iter() {
            let mut vector = vec![1.0f32; *dims];
            let length = 10.0;

            group.bench_with_input(BenchmarkId::new("Baseline", dims), dims, |b, _| {
                b.iter(|| black_box(baseline_norm_avx(vector.clone(), length)))
            });

            group.bench_with_input(BenchmarkId::new("Proposed", dims), dims, |b, _| {
                b.iter(|| unsafe { optimized_norm_avx(black_box(&mut vector), length) })
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
