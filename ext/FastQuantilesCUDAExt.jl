module FastQuantilesCUDAExt

# CUDA method of `fast_quantile` and its supporting kernels.  The shared
# driver `_fast_quantile_impl` and the selection machinery live in
# `src/fastquantile.jl`.

import FastQuantiles: fast_quantile, _fast_quantile_impl, _select_eltypes,
                      _monokey, _task_bins, _SelectTask, _refine!, _keybounds,
                      _quantile_ranks, _quantile_interp

using CUDA: CuArray, CuMatrix, CuDeviceArray, CuDeviceMatrix, CuDeviceVector,
            CuVector, CuStaticSharedArray, @cuda, @atomic, blockIdx,
            threadIdx, blockDim, gridDim, sync_threads

"""
    fast_quantile(data::CuArray, ps)
    fast_quantile(data::CuMatrix, chans_per_band, ps)

CUDA methods of `fast_quantile`: the required order statistics are selected
on the device by iterative histogram refinement (the same algorithm as the
host method in `src/fastquantile.jl`), so the data never leaves the GPU and
no sorted copy is materialized.  Each pass bins the sign-flipped IEEE bit
patterns of the values into 2048 shared-memory histogram bins per open task
and merges the block-local counts into a device histogram; the host drives
the task refinement on the few-KB histograms.  The results match
`Statistics.quantile` exactly.  The banded form (with `chans_per_band`)
selects every band's quantiles in the same fixed handful of batched passes,
so its cost is independent of the number of bands.
"""
fast_quantile(data::CuArray{<:_select_eltypes}, ps) =
    _fast_quantile_impl(_cuda_histpass!, data, ps)

# One refinement pass on the device: histogram the keys of every open task
# (block-local shared-memory histograms merged into one global histogram
# per task) and download the small counts into the reusable host-side
# `hists` workspace.  NaNs are counted in a side flag and reported by the
# host when requested.  (Device histogram buffers are per pass; the CUDA
# memory pool absorbs their allocation.)
function _cuda_histpass!(data::CuArray, tasks::Vector{_SelectTask},
                         checknan::Bool, hists::Vector{Vector{Int}})
    while length(hists) < length(tasks)
        push!(hists, zeros(Int, 2048))
    end
    nanflag = CuVector{Int32}(undef, 1)
    fill!(nanflag, Int32(0))
    threads = 256
    blocks = min(cld(length(data), threads), 1 << 15)
    for (t, task) in enumerate(tasks)
        nbins, shift = _task_bins(task)
        hist = CuVector{Int}(undef, nbins)
        fill!(hist, 0)
        @cuda threads = threads blocks = blocks _khist!(hist, nanflag, data,
                                                        task.klo, task.khi,
                                                        shift, nbins)
        copyto!(hists[t], 1, hist, 1, nbins)
    end
    if checknan
        Array(nanflag)[1] > 0 && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
    end
    return hists
end

function _khist!(hist::CuDeviceVector{Int}, nanflag::CuDeviceVector{Int32},
                 data::CuDeviceArray{<:_select_eltypes},
                 klo::UInt64, khi::UInt64, shift::Int, nbins::Int)
    shmem = CuStaticSharedArray(Int32, 2048)
    tid = Int64(threadIdx().x)
    bsz = Int64(blockDim().x)
    n = Int64(length(data))
    i = tid
    while i ≤ nbins
        shmem[i] = Int32(0)
        i += bsz
    end
    sync_threads()
    stride = bsz * Int64(gridDim().x)
    i = (Int64(blockIdx().x) - 1) * bsz + tid
    while i ≤ n
        @inbounds x = data[i]
        if x != x
            @atomic nanflag[1] += Int32(1)
        else
            key = _monokey(x)
            if klo ≤ key ≤ khi
                @atomic shmem[1 + Int((key - klo) >>> shift)] += Int32(1)
            end
        end
        i += stride
    end
    sync_threads()
    i = tid
    while i ≤ nbins
        v = shmem[i]
        if v != 0
            @atomic hist[i] += Int64(v)
        end
        i += bsz
    end
    return nothing
end

# Banded selection: every band's quantiles in the same handful of passes.
# One open `_SelectTask` per band holds all requested (offset) ranks; the
# tasks proceed through the same `_refine!` machinery as the scalar path in
# exact lockstep, since every task starts from the same full-range key
# interval and the number of passes depends only on the eltype's key width.
#
# The per-band rank arithmetic uses an offset trick so that the shared
# `_refine!` needs no modification: band `b`'s ranks and `below` count are
# offset by `(b - 1) * n_b` (with `n_b = cpb * Nr` the band size).  The
# offset cancels in every comparison inside `_refine!` (which only involves
# `below`, the per-band histogram counts, and the ranks), the resolved rank
# keys become globally unique per (band, rank) pair, and the band index is
# recoverable from a task as `below ÷ n_b`.
function fast_quantile(data::CuMatrix{T}, chans_per_band::Integer,
                       ps::Union{Real, AbstractVector, Tuple}) where {T <: _select_eltypes}
    cpb = Int(chans_per_band)
    cpb >= 1 ||
        throw(ArgumentError("chans_per_band must be at least 1 (got $cpb)"))
    n = size(data, 1)
    n > 0 || throw(ArgumentError("empty data"))
    n % cpb == 0 || throw(ArgumentError(
        "chans_per_band (= $cpb) must evenly divide the number of rows ($n)"))
    nbands = n ÷ cpb
    psv = ps isa AbstractVector ? ps :
          ps isa Tuple ? collect(ps) : [ps]
    isempty(psv) && return [zeros(promote_type(T, eltype(psv)), 0)
                            for _ in 1:nbands]
    n_b = cpb * size(data, 2)
    js, γs, ranks = _quantile_ranks(n_b, psv)
    klo0, khi0 = _keybounds(T)
    # `_refine!` walks the ranks in ascending order, so deduplicate and sort
    # once (all bands share the same local ranks).
    rr = sort!(unique(ranks))
    tasks = [_SelectTask(klo0, khi0, off, rr .+ off)
             for off in ((b - 1) * n_b for b in 1:nbands)]
    hists = Vector{Vector{Int}}()
    resolved = Dict{Int, T}()
    checknan = true
    while !isempty(tasks)
        _cuda_banded_histpass!(data, tasks, checknan, hists, cpb)
        checknan = false
        tasks = _refine!(resolved, tasks, hists)
    end
    # Per-band interpolation, identical to `_fast_quantile_impl`'s.
    out = Vector{Vector{promote_type(T, eltype(psv))}}(undef, nbands)
    for b in 1:nbands
        off = (b - 1) * n_b
        if n_b == 1
            v = resolved[off + 1]
            out[b] = [_quantile_interp(v, v, γ) for γ in γs]
        else
            out[b] = [_quantile_interp(resolved[off + js[i]],
                                       resolved[off + js[i] + 1], γs[i])
                      for i in eachindex(js)]
        end
    end
    return [ps isa Real ? only(out[b]) : ps isa Tuple ? Tuple(out[b]) : out[b]
            for b in 1:nbands]
end

# One batched refinement pass for the per-band selection tasks: a single
# kernel launch histograms every open task's band of rows — `gridDim.y` =
# number of open tasks, each block histogramming into 2048 shared-memory
# bins and merging into its task's column of the global histogram matrix —
# and downloads all histograms in one transfer into the reusable host-side
# `hists` workspace.  Each task's band of rows is recovered from its offset
# `below` count (see `fast_quantile`'s banded method).
function _cuda_banded_histpass!(fdr::CuMatrix{<:_select_eltypes},
                                tasks::Vector{_SelectTask}, checknan::Bool,
                                hists::Vector{Vector{Int}}, cpb::Int)
    while length(hists) < length(tasks)
        push!(hists, zeros(Int, 2048))
    end
    n_b = cpb * size(fdr, 2)
    bins = [_task_bins(task) for task in tasks]
    klos = CuArray(UInt64[task.klo for task in tasks])
    khis = CuArray(UInt64[task.khi for task in tasks])
    nbins = CuArray(Int[nb for (nb, _) in bins])
    shifts = CuArray(Int[sh for (_, sh) in bins])
    rows0 = CuArray(Int[(task.below ÷ n_b) * cpb + 1 for task in tasks])
    nanflag = CuVector{Int32}(undef, 1)
    fill!(nanflag, Int32(0))
    hist_dev = CuMatrix{Int}(undef, 2048, length(tasks))
    fill!(hist_dev, 0)
    threads = 256
    # Share a fixed total-block budget across the open tasks (like the
    # scalar kernel's per-task cap): more blocks would multiply the global
    # histogram merge atomics with the number of bands, while the
    # grid-stride loop simply gives each thread more elements.
    blocks_x = min(cld(n_b, threads), max(1, (1 << 15) ÷ length(tasks)))
    @cuda threads = threads blocks = (blocks_x, length(tasks)) _kbanded_hist!(
        hist_dev, nanflag, fdr, klos, khis, shifts, nbins, rows0, cpb, n_b)
    hdev = Array(hist_dev)
    for (t, (nb, _)) in enumerate(bins)
        copyto!(hists[t], 1, view(hdev, 1:nb, t), 1, nb)
    end
    if checknan
        Array(nanflag)[1] > 0 && throw(ArgumentError(
            "quantiles are undefined in presence of NaNs or missing values"))
    end
    return hists
end

# Batched variant of `_khist!`: `blockIdx().y` selects the open task, whose
# band of `cpb` matrix rows starting at `rows0[task]` (over all matrix
# columns, `n_b = cpb * Nr` elements) is histogrammed within the task's
# current key interval.
function _kbanded_hist!(hists::CuDeviceMatrix{Int},
                        nanflag::CuDeviceVector{Int32},
                        fdr::CuDeviceMatrix{<:_select_eltypes},
                        klos::CuDeviceVector{UInt64},
                        khis::CuDeviceVector{UInt64}, shifts::CuDeviceVector{Int},
                        nbins::CuDeviceVector{Int}, rows0::CuDeviceVector{Int},
                        cpb::Int, n_b::Int)
    shmem = CuStaticSharedArray(Int32, 2048)
    task = Int(blockIdx().y)
    klo = klos[task]
    khi = khis[task]
    shift = shifts[task]
    nb = nbins[task]
    row0 = rows0[task]
    tid = Int64(threadIdx().x)
    bsz = Int64(blockDim().x)
    i = tid
    while i ≤ nb
        shmem[i] = Int32(0)
        i += bsz
    end
    sync_threads()
    stride = bsz * Int64(gridDim().x)
    i = (Int64(blockIdx().x) - 1) * bsz + tid
    while i ≤ n_b
        # Linear index within the band, column-major over (cpb, Nr): the
        # decomposition keeps consecutive threads on consecutive addresses.
        r = (i - 1) % cpb + 1
        c = (i - 1) ÷ cpb + 1
        @inbounds x = fdr[row0 + r - 1, c]
        if x != x
            @atomic nanflag[1] += Int32(1)
        else
            key = _monokey(x)
            if klo ≤ key ≤ khi
                @atomic shmem[1 + Int((key - klo) >>> shift)] += Int32(1)
            end
        end
        i += stride
    end
    sync_threads()
    i = tid
    while i ≤ nb
        v = shmem[i]
        if v != 0
            @atomic hists[i, task] += Int64(v)
        end
        i += bsz
    end
    return nothing
end

end # module FastQuantilesCUDAExt
