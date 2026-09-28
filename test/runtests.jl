using Test
using Statistics
using Random
using FastQuantiles

@testset "fast_quantile" begin
    # The histogram-refinement selection must reproduce
    # `Statistics.quantile` exactly, including types
    qsel(x, ps) = fast_quantile(x, ps)
    qrng = MersenneTwister(7)
    for T in (Float32, Float64, Int64, Int32, Float16, UInt16)
        for n in (1, 2, 3, 7, 100, 1000)
            for kind in
                (T <: Integer ? (:rand, :ties, :ints) : (:rand, :randn, :ties))
                x = if kind === :rand
                    rand(qrng, T, n)
                elseif kind === :randn
                    randn(qrng, T, n)
                elseif kind === :ties
                    T.(round.(rand(qrng, Float64, n) .* 5))
                else
                    rand(qrng, 1:10, n) .% T
                end
                for ps in ([0.1, 0.5], [0.0, 0.05, 0.3, 0.5, 0.999, 1.0])
                    @test qsel(x, ps) == quantile(vec(x), ps)
                    @test typeof(qsel(x, ps)) == typeof(quantile(vec(x), ps))
                end
            end
        end
    end
    # Degenerate and extreme values
    for x in (fill(3.0, 100), [Inf, 1.0, -Inf, 2.0], rand(qrng, 1:10, 500),
              fill(Int32(7), 3), [-0.0, 0.0, 1.5],
              [typemax(Int32), typemin(Int32), Int32(0)],
              [1.0f0, 1.0f0 + eps(1.0f0), 1.0f0 - eps(1.0f0)],
              reshape(collect(Float32, 1.0:12.0), 3, 4))
        for ps in ([0.1, 0.5], [0.0, 0.5, 1.0])
            @test qsel(x, ps) == quantile(vec(x), ps)
        end
    end
    # Scalar and tuple probability forms
    x = collect(1.0:100)
    @test qsel(x, 0.25) == quantile(x, 0.25)
    @test qsel(x, (0.1, 0.5)) == quantile(x, (0.1, 0.5))
    # NaN handling matches `quantile`
    @test_throws ArgumentError qsel([1.0, NaN, 3.0], [0.5])
    @test_throws ArgumentError quantile([1.0, NaN, 3.0], [0.5])
    # Large-array spot check (Float32 keys resolve in few passes)
    x = randexp(qrng, Float32, 2_000_000)
    @test qsel(x, [0.1, 0.5]) == quantile(vec(x), [0.1, 0.5])
    # Unsupported eltypes fall back to `quantile`
    xr = Rational{Int}.(1:20, 21)
    @test fast_quantile(xr, [0.1, 0.5]) == quantile(xr, [0.1, 0.5])

    # Banded selection: each band's result is exactly the scalar result for
    # that band's rows
    rngb = MersenneTwister(19)
    xb = randexp(rngb, Float32, 24, 32)
    for ps in (0.1, (0.1, 0.5), [0.1, 0.5])
        banded = fast_quantile(xb, 8, ps)
        @test length(banded) == 3
        ref_ps = ps isa Real ? [ps] : ps isa Tuple ? collect(ps) : ps
        for b in 1:3
            band = xb[((b - 1) * 8 + 1):(b * 8), :]
            @test banded[b] == fast_quantile(band, ps)
            ref = quantile(vec(band), ref_ps)
            @test (ps isa Real ? [banded[b]] :
                   ps isa Tuple ? collect(banded[b]) : banded[b]) == ref
        end
    end
    @test fast_quantile(xb, 8, 0.1) isa Vector{Float64}
    @test fast_quantile(xb, 8, (0.1, 0.5)) isa Vector{<:Tuple}
    # Single-row bands and NaN rejection
    @test fast_quantile(xb, 1, 0.5) ==
          [quantile(vec(xb[i, :]), 0.5) for i in 1:size(xb, 1)]
    @test_throws ArgumentError fast_quantile([1.0f0 NaN32; 3.0f0 4.0f0], 1, 0.5)
    # Errors: non-positive and non-divisor band widths, empty data
    @test_throws ArgumentError fast_quantile(xb, 0, 0.5)
    @test_throws ArgumentError fast_quantile(xb, 5, 0.5)
    @test_throws ArgumentError fast_quantile(zeros(0, 4), 2, 0.5)
end

using CUDA

if CUDA.functional()
    @testset "fast_quantile [CUDA]" begin
        rngg = MersenneTwister(11)
        gm2 = randexp(rngg, Float32, 256, 256) .+
              randexp(rngg, Float32, 256, 256)
        # On-device selection matches `quantile` exactly
        @test fast_quantile(CuArray(gm2), [0.1, 0.5]) ==
              quantile(vec(gm2), [0.1, 0.5])
        gi = Int32.(round.(gm2 .* 100))
        @test fast_quantile(CuArray(gi), [0.25, 0.75]) ==
              quantile(vec(gi), [0.25, 0.75])
        @test_throws ArgumentError fast_quantile(
            CuArray([1.0f0, NaN32, 3.0f0]), [0.5])
        # Non-finite values (keys are bit patterns, so Inf works)
        @test fast_quantile(CuArray([Inf32, 1.0f0, -Inf32, 2.0f0]),
                            [0.0, 0.5, 1.0]) ==
              quantile([Inf32, 1.0f0, -Inf32, 2.0f0], [0.0, 0.5, 1.0])

        # Banded selection on the device matches the host bit-for-bit
        rngb = MersenneTwister(19)
        xb = randexp(rngb, Float32, 24, 32)
        xbd = CuArray(xb)
        for ps in (0.1, (0.1, 0.5), [0.1, 0.5])
            @test fast_quantile(xbd, 8, ps) == fast_quantile(xb, 8, ps)
        end
        # Bitwise reproducible across calls
        @test fast_quantile(xbd, 8, [0.1, 0.5]) == fast_quantile(xbd, 8, [0.1, 0.5])
        # Single-row bands, NaN rejection, and divisibility errors
        @test fast_quantile(xbd, 1, 0.5) == fast_quantile(xb, 1, 0.5)
        @test_throws ArgumentError fast_quantile(
            CuArray([1.0f0 NaN32; 3.0f0 4.0f0]), 1, 0.5)
        @test_throws ArgumentError fast_quantile(xbd, 5, 0.5)
    end
else
    @info "Skipping CUDA tests: no functional GPU available"
end
