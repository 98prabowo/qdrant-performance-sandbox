use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(target_arch = "aarch64")]
mod neon_bench {
    use super::*;
    use qdrant_performance_sandbox::hsum_neon::{baseline_sum, optimized_sum};
    use std::arch::aarch64::*;

    pub fn run(c: &mut Criterion) {
        let mut group = c.benchmark_group("Horizontal Sum/Neon");
        group.warm_up_time(Duration::from_secs(5));
        group.sample_size(200);

        unsafe {
            let sum1 = vdupq_n_f32(1.1);
            let sum2 = vdupq_n_f32(2.2);
            let sum3 = vdupq_n_f32(3.3);
            let sum4 = vdupq_n_f32(4.4);

            group.bench_function("Baseline", |b| {
                b.iter(|| black_box(baseline_sum(sum1, sum2, sum3, sum4)));
            });

            group.bench_function("Proposed", |b| {
                b.iter(|| black_box(optimized_sum(sum1, sum2, sum3, sum4)));
            });
        }

        group.finish();
    }
}

fn run_benchmarks(c: &mut Criterion) {
    #[cfg(target_arch = "aarch64")]
    neon_bench::run(c);
}

criterion_group!(benches, run_benchmarks);
criterion_main!(benches);
