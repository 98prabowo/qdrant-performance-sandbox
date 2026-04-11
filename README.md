# Qdrant Performance Lab 🧪

A specialized benchmarking and research environment for high-performance systems programming in Rust.
This lab is used to prototype, profile, and verify low-level optimizations before they are integrated into production-grade systems like [Qdrant](https://github.com/qdrant/qdrant).

## Performance Results

The following results represent the latest stable benchmarks for the aarch64 NEON optimizations.

| Module             | Benchmark     | Baseline  | Proposed      | Improvement |
| ------------------ | ------------- | --------- | ------------- | ----------- |
| **Horizontal Sum** | 200 samples   | 1.73 ns   | **1.57 ns**   | ~9.2%       |
| **Normalization**  | Small (1536d) | 234.99 ns | **188.71 ns** | ~19.7%      |
| **Normalization**  | Big (1M dims) | 88.83 µs  | **77.81 µs**  | ~12.4%      |

### Environment:

- **Hardware:** Apple M1 (16GB RAM)
- **Architecture:** `aarch64-apple-darwin`
- **Compiler:** `rustc 1.94.1`

## Purpose

The goal of this project is to provide a "clean room" for performance engineering.
By isolating core algorithms from the overhead of a full database engine, we can:

- **Quantify** the exact latency impact of micro-optimizations.
- **Verify** memory safety and architectural constraints (SIMD, Cache alignment, etc.).
- **Compare** multiple implementation strategies (e.g., Scalar vs. Auto-vectorized vs. Intrinsics).

## Project Architecture

The lab follows a decoupled structure to ensure that implementation logic remains separate from measurement logic, facilitating easy migration of code into production targets.

- `src/`: Contains core logic, algorithm implementations, safety documentation, and unit tests.
- `benches/`: Contains the [criterion.rs](https://github.com/bheisler/criterion.rs) measurement suites.
- `reports/`: Contains latest benchmarks reports.

## Current Research Modules

### 1. Vector Normalization (`normalization.rs`)

Focuses on the `cosine_preprocess` kernel. This module explores efficient ways to normalize high-dimensional vectors.

- **Techniques:** In-place mutation to eliminate allocations, loop unrolling, and manual SIMD (NEON).
- **Target:** Reducing preprocessing latency in vector similarity searches.

### 2. Horizontal Reductions (`hsum.rs`)

A micro-benchmark suite focused on the reduction phase of floating-point operations.

**Techniques:** Vertical pairing vs. horizontal accumulation to minimize CPU pipeline stalls.

## Benchmarking Methodology

We use **Criterion** to ensure statistical rigor. All benchmarks account for:

- **CPU Warmup:** Ensuring the processor is at peak frequency before measuring.
- **Outlier Detection:** Identifying OS-level interruptions that could skew data.
- **Confidence Intervals:** Providing a range of expected performance rather than a single number.
- **Thermal Management:** Integrated `sleep` intervals between iterations to minimize frequency scaling (throttling) effects during sustained SIMD loads.

To run the suite:

```sh
# Benchmark neon hsum
make bench-hsum-neon

# Benchmark neon normalization
make bench-norm-neon
```

## Remote Benchmarking (GitHub Actions)

You can verify these results on standardized cloud hardware using GitHub Actions.
This project is configured to run on native **ARM64 (Ubuntu 24.04)** runners, providing a neutral environment (Ampere/Neoverse cores) to compare against local Apple Silicon results.

### How to run:

1. **Fork** this repository.
2. Go to the **Actions** tab in your fork.
3. Select the **"ARM64 Performance Benchmark"** workflow on the left sidebar.
4. Click the **Run workflow** dropdown and select the `main` branch.
5. Once complete, click on the run to view the logs.

The output will display a side-by-side comparison of the **Baseline** vs. **Proposed** implementations, including statistical analysis of the speedup.

> [!TIP]
> This is the preferred way to verify PRs, as it eliminates "noise" from local background processes and thermal throttling common on laptop environments.

## License

This laboratory is licensed under the **Apache License 2.0**. This allows for seamless code sharing with the Qdrant core and other high-performance open-source projects.
