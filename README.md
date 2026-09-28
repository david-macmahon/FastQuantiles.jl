# FastQuantiles.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://david-macmahon.github.io/FastQuantiles.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://david-macmahon.github.io/FastQuantiles.jl/dev/)
[![Build Status](https://github.com/david-macmahon/FastQuantiles.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/david-macmahon/FastQuantiles.jl/actions/workflows/CI.yml)

Exact quantiles of large arrays, without sorting.

`fast_quantile(data, ps)` computes quantiles that match
`Statistics.quantile(vec(data), ps)` **bit-for-bit** (same rank arithmetic,
interpolation, NaN handling, and result types), but it neither copies nor
sorts the data.  For large arrays this is dramatically faster:
`Statistics.quantile` single-threadedly partially sorts a copy of the data
(about 95 s for a 4 GiB `Float32` array), while `fast_quantile` makes a
fixed handful of streaming passes over the data (~3 s multithreaded on the
host, ~0.07 s on an NVIDIA GPU).

## Usage

```julia
julia> using FastQuantiles

julia> data = randexp(Float32, 100_000_000);   # ~400 MB

julia> fast_quantile(data, [0.1, 0.5])
2-element Vector{Float64}:
 0.104...
 0.693...

julia> fast_quantile(data, 0.5)        # scalar probability gives a scalar
0.693...

julia> fast_quantile(data, (0.1, 0.9)) # tuple probability gives a tuple
(0.104..., 1.61...)
```

Arrays on an NVIDIA GPU work the same way (the CUDA extension runs the
same algorithm on the device, so the data never leaves the GPU):

```julia
julia> using CUDA

julia> fast_quantile(CuArray(data), [0.1, 0.5])
```

Inputs with eltypes that have no order-preserving bit pattern fall back to
`Statistics.quantile`.  Supported eltypes: `Float16`, `Float32`, `Float64`,
`Bool`, and the fixed-size signed/unsigned integer types.

Matrices can also be treated as consecutive *bands* of rows:
`fast_quantile(data, chans_per_band, ps)` returns one result per band
(each exactly what `fast_quantile` returns for that band's rows).  On the
host each band is selected independently; on CUDA all bands are selected
in the same fixed handful of batched passes, at a cost independent of the
number of bands.

## Installation

The package is not yet registered; install directly from the repository:

```julia
julia> using Pkg

julia> Pkg.add(url = "https://github.com/david-macmahon/FastQuantiles.jl")
```

## How it works (briefly)

The handful of order statistics that the quantile definition needs are
selected exactly by *iterative histogram refinement*: values are remapped
to unsigned integer keys that preserve their total order (the sign-flipped
IEEE bit patterns for floats), the key interval containing each required
rank is narrowed by binning into 2048 bins per pass, and a bin holding a
single key resolves every rank it contains.  Ties are just duplicate keys
and resolve naturally.  See the
[documentation](https://david-macmahon.github.io/FastQuantiles.jl) for the
full theory of operation.

## License

This package is licensed under the [BSD 2-Clause "Simplified"
License](LICENSE).
