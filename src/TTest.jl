module TTest
export TTestObj

include("Utils.jl")
using .Utils
include("Moments.jl")
using .Moments

# Can't be named TTest because thats already what the module is named 
mutable struct TTestObj{Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    accumulators::Vector{UniVarMomentsAcc{Tt, Tl, Tarray}}
    const order::UInt
    const ns::UInt
    const nl::UInt
    const chunk_size::NTuple{2, Int}

    function TTestObj{Tt, Tl, Tarray}(order, ns, nl, chunk_size::NTuple{2}) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
        # The accumulators all expect a full size chunk.
        accumulators = [UniVarMomentsAcc{Tt, Tl, Tarray}(order, chunk_size[2], nl) for _ in 1:cld(ns, chunk_size[2])]
        new(accumulators, order, ns, nl, chunk_size)
    end
end

function ttest_fit(ttest::TTestObj{Tt, Tl, Tarray}, traces, labels) where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    trace_tiles, trace_indices = tiled_view(traces, ttest.chunk_size, return_indices=true)
    label_tiles, label_indices = tiled_view(labels, (ttest.chunk_size[1], ), return_indices=true)
    
    for chunk in axes(traces, 1)
        for yslice in axes(trace_tiles, 2)
            res = centered_sum_update!(ttest.accumulators[chunk], trace_tiles[chunk, yslice], label_tiles[chunk])
            # I need to figure out how to run the whole centered_sum_update inplace
        end
    end
end

function ttest_finalize(acc::UniVarMomentsAcc{Tt, Tl, Tarray}, order::Int)::Vector where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    acc = 
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

function ttest_finalize(ttest::TTestObj{Tt, Tl, Tarray})::Vector where {Tt<:AbstractFloat, Tl<:Integer, Tarray<:AbstractArray}
    result = vcat(ttest_finalize.(ttest.accumulators, ttest.order))
end



end  # module TTest