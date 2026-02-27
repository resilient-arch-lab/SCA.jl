module TTest
export ttest_fit!, ttest_finalize, TTestSingle, TTestChunked

include("Utils.jl")
using .Utils
include("Moments.jl")
using .Moments

mutable struct TTestSingle{Tt<:AbstractFloat, Tl<:Integer}
    moments::UniVarMomentsAcc{Tt, Tl, Array}
    order::Int
    ns::Int

    function TTestSingle{Tt, Tl}(order::Int, ns::Int) where {Tt<:AbstractFloat, Tl<:Integer}
        moments = UniVarMomentsAcc{Tt, Tl, Array}(2*order, ns, 2)
        new(moments, order, ns)
    end
end

mutable struct TTestChunked{Tt<:AbstractFloat, Tl<:Integer}
    chunksize::NTuple{2, Int}
    chunk_map::Dict{UnitRange, TTestSingle}
    order::Int
    ns::Int

    function TTestChunked{Tt, Tl}(order::Int, ns::Int, chunksize::NTuple{2, Int}) where {Tt<:AbstractFloat, Tl<:Integer}
        slices = tiled_view(1:ns, (chunksize[2], ))
        chunk_map = Dict(slice => TTestSingle{Tt, Tl}(order, size(slice, 1)) for slice in slices)
        new(chunksize, chunk_map, order, ns)
    end
end

function ttest_fit!(ttest::TTestSingle{Tt, Tl}, traces, labels) where {Tt<:AbstractFloat, Tl<:Integer}
    centered_sum_update!(ttest.moments, traces, labels)
end

function ttest_fit!(ttest::TTestChunked{Tt, Tl}, traces, labels) where {Tt<:AbstractFloat, Tl<:Integer}
    (trace_tiles, tile_indices) = tiled_view(traces, ttest.chunksize; return_indices=true)
    label_tiles = tiled_view(labels, (ttest.chunksize[1], ))

    for (t_tile, l_tile, tile_idx) in zip(trace_tiles, repeat(label_tiles, outer=(1, size(trace_tiles, 2))), tile_indices)
        # NOTE: tiles which don't overlap on dimension 2 can be processed concurrently. 
        ttest_fit!(ttest.chunk_map[tile_idx[2]], t_tile, l_tile)
    end
end

function ttest_finalize(ttest::TTestSingle{Tt, Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    μ, σ = get_mean_and_var(ttest.moments, ttest.order)
    μ1, μ2 = view(μ, 1, :), view(μ, 2, :) 
    σ1, σ2 = view(σ, 1, :), view(σ, 2, :) 
    t = (μ1 - μ2) ./ sqrt.((σ1 ./ ttest.moments.totals[1]) .+ (σ2 ./ ttest.moments.totals[2]))
end

function ttest_finalize(ttest::TTestChunked{Tt, Tl}) where {Tt<:AbstractFloat, Tl<:Integer}
    out = zeros(ttest.ns)
    Threads.@threads for slice in collect(keys(ttest.chunk_map))
        out[slice] .= ttest_finalize(ttest.chunk_map[slice])
    end
    out
end

function ttest_finalize(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, order::Int)::Vector where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    if acc.nl != 2
        error("TTest must be performed with UniVarMomentsAcc with nl=2, got nl=$(acc.nl)")
    end

    t = Tarray{Tt, 1}(undef, acc.ns)

    if order == 1
        @inbounds u1, u2 = acc.moments[1, 1, :], acc.moments[2, 1, :]
        @inbounds v1, v2 = acc.moments[1, 2, :] ./ acc.totals[1], acc.moments[2, 2, :] ./ acc.totals[2]

        @inbounds t .= (u1 - u2) ./ sqrt.((v1 ./ acc.totals[1]) .+ (v2 ./ acc.totals[2]))
        return t
    elseif order == 2
        @inbounds u1, u2 = acc.moments[1, 2, :] ./ acc.totals[1], acc.moments[2, 2, :] ./ acc.totals[2]
        @inbounds v1, v2 = acc.moments[1, 4, :] ./ acc.totals[1], acc.moments[2, 4, :] ./ acc.totals[2]

        @inbounds v1 .-= (u1.^2)
        @inbounds v2 .-= (u2.^2)

        @inbounds t .= (u1 - u2) ./ sqrt.((v1 ./ acc.totals[1]) .+ (v2 ./ acc.totals[2]))
        return t
    else
        @inbounds u1 = (acc.moments[1, order, :] ./ acc.totals[1]) ./ ((acc.moments[1, 2, :] ./ acc.totals[1]).^(order/2))
        @inbounds u2 = (acc.moments[2, order, :] ./ acc.totals[2]) ./ ((acc.moments[2, 2, :] ./ acc.totals[2]).^(order/2))

        @inbounds v1 = ((acc.moments[1, 2*order, :] ./ acc.totals[1]) .- ((acc.moments[1, order, :] ./ acc.totals[1]).^2)) ./ ((acc.moments[1, 2, :] ./ acc.totals[1]).^order)
        @inbounds v2 = ((acc.moments[2, 2*order, :] ./ acc.totals[2]) .- ((acc.moments[2, order, :] ./ acc.totals[2]).^2)) ./ ((acc.moments[2, 2, :] ./ acc.totals[2]).^order)

        @inbounds t .= (u1 - u2) ./ sqrt.((v1 ./ acc.totals[1]) .+ (v2 ./ acc.totals[2]))
        return t
    end
end



end  # module TTest