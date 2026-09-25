# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-09-25

### Added

- `fast_quantile(data, ps)` computing quantiles of floating-point arrays
  bit-for-bit compatible with `Statistics.quantile`, without copying or
  sorting the data, via iterative histogram refinement.
- Support for scalar, vector, and tuple probability inputs, with matching
  scalar/vector/tuple result types.
- Supported eltypes: `Float16`, `Float32`, `Float64`, `Bool`, and fixed-size
  signed/unsigned integers; other eltypes fall back to `Statistics.quantile`.
- Optional CUDA extension (`FastQuantilesCUDAExt`) running the same algorithm
  on-device for `CuArray` inputs, so the data never leaves the GPU.
- Multithreaded CPU implementation for large arrays.
- Documentation, CI, and docs-deployment GitHub Actions workflows.

[Unreleased]: https://github.com/david-macmahon/FastQuantiles.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/david-macmahon/FastQuantiles.jl/releases/tag/v0.1.0
