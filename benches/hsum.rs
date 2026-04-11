use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
mod neon_bench {
    use super::*;
    use criterion::BatchSize;
    use qdrant_performance_sandbox::hsum_neon::{baseline_sum, optimized_sum};
    use std::arch::aarch64::*;

    pub fn run(c: &mut Criterion) {
        let mut group = c.benchmark_group("Horizontal Sum/Neon");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        let data = [
            1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.1, 11.2, 12.3, 13.4, 14.5, 15.6, 16.7,
        ];

        group.bench_function("Baseline", |b| {
            b.iter_batched(
                || unsafe {
                    (
                        vld1q_f32(data.as_ptr()),
                        vld1q_f32(data.as_ptr().add(4)),
                        vld1q_f32(data.as_ptr().add(8)),
                        vld1q_f32(data.as_ptr().add(12)),
                    )
                },
                |(sum1, sum2, sum3, sum4)| unsafe {
                    black_box(baseline_sum(sum1, sum2, sum3, sum4))
                },
                BatchSize::SmallInput,
            );
        });

        group.bench_function("Proposed", |b| {
            b.iter_batched(
                || unsafe {
                    (
                        vld1q_f32(data.as_ptr()),
                        vld1q_f32(data.as_ptr().add(4)),
                        vld1q_f32(data.as_ptr().add(8)),
                        vld1q_f32(data.as_ptr().add(12)),
                    )
                },
                |(sum1, sum2, sum3, sum4)| unsafe {
                    black_box(optimized_sum(sum1, sum2, sum3, sum4))
                },
                BatchSize::SmallInput,
            );
        });

        group.finish();
    }
}

fn run_benchmarks(c: &mut Criterion) {
    #[cfg(target_arch = "aarch64")]
    neon_bench::run(c);
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
