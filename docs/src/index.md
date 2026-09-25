# FastQuantiles.jl

Exact quantiles of large arrays, without sorting.  `fast_quantile(data, ps)`
computes quantiles that match `Statistics.quantile(vec(data), ps)`
**bit-for-bit** (the same rank arithmetic, interpolation, NaN handling, and
result types; verified by the test suite) but it neither copies nor sorts
the data, which makes it much faster for large arrays:

| 4 GiB `Float32`, 2 quantiles | time |
|---|---|
| `Statistics.quantile` (host) | ~95 s |
| `fast_quantile` (host, 16 threads) | ~3 s |
| `fast_quantile` (NVIDIA GPU) | ~0.07 s |

## Usage

```julia
julia> using FastQuantiles

julia> data = randexp(Float32, 100_000_000);   # ~400 MB

julia> fast_quantile(data, [0.1, 0.5])
2-element Vector{Float64}:
 0.10491534
 0.6931697

julia> fast_quantile(data, 0.5)          # scalar probability gives a scalar
0.6931697

julia> fast_quantile(data, (0.1, 0.9))   # tuple probability gives a tuple
(0.10491534, 1.6094379)
```

The `ps` argument follows the same conventions as `Statistics.quantile`: a
single probability returns a scalar, a vector returns a vector, and a tuple
returns a tuple.  Arrays on an NVIDIA GPU work the same way (the CUDA
extension runs the same algorithm on the device, so the data never leaves
the GPU:

```julia
julia> using CUDA

julia> fast_quantile(CuArray(data), [0.1, 0.5])
2-element Vector{Float64}:
 0.10491534
 0.6931697
```

On the host, the histogram passes are multithreaded when
`Threads.nthreads()` is greater than one and the data is large (at least
2^20 elements).  The GPU method needs no scratch space proportional to the
data size (unlike a device-side sort, which requires a full-size temporary
buffer), and the number of requested quantiles is nearly free since all of
them share each pass's histograms.

Inputs with eltypes that have no order-preserving bit pattern (e.g.
`Rational`, `BigInt`) fall back to `Statistics.quantile`.  Supported
eltypes are `Float16`, `Float32`, `Float64`, `Bool`, and the fixed-size
signed and unsigned integer types.

```@docs
fast_quantile
```

## Theory of operation

### Why not sort?

`Statistics.quantile` copies its input and then partially sorts that copy
with a single-threaded comparison sort (`PartialQuickSort` over the rank
range).  For a 4 GiB `Float32` array the copy is fast (~1 s), but the sort
is cache-hostile and serial: ~95 s.  A quantile only needs a handful of
*order statistics* (for two probabilities, at most four values), so
sorting is far more work than necessary.  What is needed is a *selection*
algorithm whose cost is proportional to a few streaming passes over the
data, not to a sort.

### Order-preserving integer keys

The algorithm never compares values directly; it maps each value to an
unsigned 64-bit integer *key* that preserves the value's total order:

- **Floats** (including `Float16` and `Float32`, via exact widening): the
  IEEE bit pattern, sign-flipped.  For a non-negative float `x`,
  `key = bits(x) | 0x8000...`; for a negative float, `key = ~bits(x)`.
  Because IEEE bit patterns of non-negative floats increase monotonically
  with the value and decrease monotonically for negative floats, this maps
  float order exactly onto unsigned integer order (`-Inf < -0.0 < +0.0 <
  Inf < NaN`).
- **Signed integers**: the two's-complement bit pattern with the sign bit
  flipped (`bits(x) ⊻ 0x8000...`), which maps signed order onto unsigned
  order.
- **Unsigned integers and `Bool`**: the value itself.

The map is a bijection between values and keys, so a key *is* the value:
the inverse map returns the exact original bit pattern.  `NaN` values are
detected and rejected up front (matching `quantile`, which throws
`ArgumentError`), so keys always correspond to well-ordered values.  This
removes all floating-point edge cases (subnormals, `-0.0`, `±Inf`) from
the selection logic: after the key transform, the data is just unsigned
integers.

### Merge-tree selection by histogram refinement

The ranks required by the quantile definition are known up front: for
probability `p` and `n` points, the definition (`alpha = beta = 1`, i.e.
linear interpolation, as in `Statistics.quantile`) needs the order
statistics at ranks `j` and `j + 1` where `j = trunc(fma(n, p, 1 - p))`.
All requested ranks are then selected together by *iterative histogram
refinement*:

1. **State.**  A small set of open *tasks*.  A task is a key interval
   `[klo, khi]` known to contain the elements of a specific list of
   (global, 1-based) ranks, together with `below`, the exact number of
   elements keyed below `klo`.  Initially there is a single task covering
   the full key range with all requested ranks.
2. **Histogram pass.**  Each task's key interval is partitioned into 2048
   bins by a right shift (`bin = 1 + (key - klo) >>> shift`, with
   `width < 2048` handled by giving every key its own bin).  One pass over
   the data computes, for every open task, the histogram of the keys
   falling in its interval.  Keys are order-preserving, so each bin's
   contents are contiguous in value order, and the bin partition is exact
   integer arithmetic (no rounding, no boundaries to get wrong).
3. **Refinement.**  Bins are walked in order with running prefix counts to
   locate the bin containing each rank.  A bin covering a *single* key
   resolves every rank it holds (all elements with that key are equal, so
   the quantile interpolation sees `a == b`); other bins spawn new tasks
   with narrowed intervals and updated `below` counts.  Bins partition
   tasks exactly, so sibling tasks never overlap and every element is
   counted exactly once.
4. **Termination.**  Each pass multiplies the interval width by the
   reciprocal of the bin count, so the width (measured in key bits)
   shrinks by 11 bits per pass.  A `Float32` key (32 bits) needs 3
   refinement passes; a `Float64`/`Int64` key (64 bits) needs 6.  Ties are
   simply duplicate keys and converge like any other value; massive ties
   (e.g. integer-valued data) do not stall the narrowing because the bin
   *widths* shrink geometrically regardless of the data.

The final interpolation replicates `Statistics._quantile` for its default
`alpha = beta = 1` exactly (including the `a ≈ b` fast path and the
handling of non-finite values), so results are bit-for-bit identical to
`Statistics.quantile`, which the test suite verifies across eltypes,
sizes, distributions, and probability values.

### Complexity and parallelism

- **Time**: one streaming histogram pass per refinement level,
  `ceil(bits/11)` passes in total (3 for `Float32`, 6 for `Float64`), each
  touching all `n` elements once.  With
  a single thread this is bandwidth-bound; with multiple threads
  (`Threads.@threads` over disjoint chunks with per-chunk histograms that
  are summed afterwards) it scales with memory bandwidth.
- **Space**: `O(2048)` counters per open task plus the (few) output
  values.  Nothing proportional to `n` is ever allocated (no copy, no
  sorted buffer, no gather).
- **Rank count**: requesting more quantiles costs almost nothing extra,
  since all ranks ride the same histograms until they diverge into
  different bins.

### CUDA extension

When CUDA.jl is loaded, `fast_quantile(::CuArray, ps)` runs the same
algorithm with the data resident on the device:

- Each histogram pass is a CUDA kernel: each thread block accumulates a
  2048-bin histogram in *shared memory* (fast on-chip atomics), then merges
  it into a global device histogram.  The grid uses a fixed upper bound on
  block count with a grid-stride loop, so blocks amortize the shared-memory
  zeroing over many elements.
- The host downloads only the per-task histograms (a few KiB per pass),
  performs the exact same task refinement as the host path, and uploads the
  (few) narrowed task intervals for the next pass.
- `NaN`s are counted by an atomic side flag in the first pass and reported
  by the host, matching the host path's error behavior.
- Memory: `O(2048)` counters (no full-size scratch buffer), so the method
  works for arrays far larger than a device-side sort could handle.

Because the host performs the refinement, the merge-tree logic (and its
exactness guarantees) exists in exactly one place, shared by both paths.

## License

This package is licensed under the [BSD 2-Clause "Simplified"
License](https://github.com/david-macmahon/FastQuantiles.jl/blob/main/LICENSE).
