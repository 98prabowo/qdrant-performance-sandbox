# 🧪 Qdrant Performance Lab 

A specialized benchmarking and research environment for high-performance systems programming in Rust. 
This lab is used to prototype, profile, and verify low-level optimizations before they are integrated into production-grade systems like [Qdrant](https://github.com/qdrant/qdrant).

## Purpose

The goal of this project is to provide a "clean room" for performance engineering. 
By isolating core algorithms from the overhead of a full database engine, we can:

- **Quantify** the exact latency impact of micro-optimizations.
- **Verify** memory safety and architectural constraints (SIMD, Cache alignment, etc.).
- **Compare** multiple implementation strategies (e.g., Scalar vs. Auto-vectorized vs. Intrinsics).

## Project Architecture

The lab follows a decoupled structure to ensure that implementation logic remains separate from measurement logic, facilitating easy migration of code into production targets.

- `src/`: Contains core logic, algorithm implementations, and safety documentation.
- `benches/`: Contains the [criterion.rs](https://github.com/bheisler/criterion.rs) measurement suites.
- `tests/`: Functional verification to ensure that "faster" doesn't mean "wrong."

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

To run the full suite:

```sh
make bench
```

## Safety & Documentation

Every `unsafe` implementation in this lab is required to include a `/// Safety` section. This section must detail:

1. **Memory Alignment:** Requirements for input pointers.
1. **Architecture Constraints:** Target features required (e.g., `+neon`, `+avx2`).
1. **Bounds Checking:** How the implementation prevents out-of-bounds access on remainders.

## License

This laboratory is licensed under the **Apache License 2.0**. This allows for seamless code sharing with the Qdrant core and other high-performance open-source projects.
