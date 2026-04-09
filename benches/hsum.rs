#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use std::{hint::black_box, time::Duration};

use criterion::{Criterion, criterion_group, criterion_main};
use qdrant_performance_sandbox::hsum::{baseline_sum, optimized_sum};

fn benchmark_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("Horizontal Sum");
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

criterion_group!(benches, benchmark_sum);
criterion_main!(benches);
