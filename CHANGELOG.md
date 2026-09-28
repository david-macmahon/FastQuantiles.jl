# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-28

### Added

- Banded selection: `fast_quantile(data, chans_per_band, ps)` treats a
  matrix as consecutive bands of `chans_per_band` rows and returns one
  result per band (each exactly what `fast_quantile` returns for that
  band's data).  On the host each band is selected independently; on CUDA
  all bands are selected in the same fixed handful of batched passes, so
  the cost is independent of the number of bands.  Per-band results are
  bit-for-bit identical to per-band `fast_quantile` calls on both paths.

### Changed

- The internal histogram passes now run in caller-owned reusable buffers
  (`histpass(data, tasks, checknan, hists)`), so repeated refinement
  passes and repeated selections (e.g. per band) no longer churn histogram
  allocations.  Internal change only; `fast_quantile` results are
  unaffected.

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

### Fixed

- `fast_quantile` matches `Statistics.quantile` bit-for-bit on Julia 1.10 as
  well, by replicating that version's rank arithmetic (`n*p + m`; the `fma`
  form is only used for Statistics ≥ 1.11).
- The docs deployment workflow runs `docs/make.jl` with `--project=docs/` so
  the docs environment is actually used.

[Unreleased]: https://github.com/david-macmahon/FastQuantiles.jl/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/david-macmahon/FastQuantiles.jl/releases/tag/v0.1.0
