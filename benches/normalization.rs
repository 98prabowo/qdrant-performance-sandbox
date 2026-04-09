use std::{hint::black_box, time::Duration};

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use qdrant_performance_sandbox::normalization::{
    baseline_normalization, neon_normalization_optimized,
};

const SMALL_DIMS: usize = 1532; // Standard AI vector size (e.g., OpenAI embeddings)
const BIG_DIMS: usize = 1_000_000;

fn benchmark_normalization(c: &mut Criterion) {
    let small_vector_base = vec![0.5f32; SMALL_DIMS];
    let big_vector_base = vec![0.5f32; BIG_DIMS];
    let length = 10.0f32; // Pre-calculated length for the benchmark

    let mut group = c.benchmark_group("Normalization");
    group.warm_up_time(Duration::from_secs(5));
    group.sample_size(200);

    group.bench_function("Small/Baseline", |b| {
        b.iter(|| {
            let input = black_box(small_vector_base.clone());
            baseline_normalization(input, length);
        });
    });

    group.bench_function("Small/Proposed", |b| {
        b.iter(|| {
            let mut input = black_box(small_vector_base.clone());
            unsafe {
                neon_normalization_optimized(&mut input, length);
            }
        });
    });

    group.bench_function("Big/Baseline", |b| {
        b.iter_batched(
            || big_vector_base.clone(),
            |input| baseline_normalization(input, length),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("Big/Proposed", |b| {
        b.iter_batched(
            || big_vector_base.clone(),
            |mut input| unsafe { neon_normalization_optimized(&mut input, length) },
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, benchmark_normalization);
criterion_main!(benches);
