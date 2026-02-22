"""
Signal to Noise Ratio (SNR)
"""

module SNR

include("Utils.jl")
using .Utils
include("Moments.jl")
using .Moments

using Statistics
using Atomix

mutable struct SNRBasic{Tt<:AbstractFloat, Tl<:Integer}
    sums::AbstractMatrix{Tt}
    sums_sq::AbstractMatrix{Tt}
    totals::AbstractVector{Int}
    const nl::Int
    const ns::Int

    function SNRBasic{Tt, Tl}(ns::Int, nl::Int) where {Tt<:AbstractFloat, Tl<:Integer}
        sums = zeros(Tt, nl, ns)
        sums_sq = zeros(Tt, nl, ns)
        totals = zeros(Tl, nl)
        new(sums, sums_sq, totals, nl, ns)
    end
end

mutable struct SNRMoments{Tt<:AbstractFloat, Tl<:Integer}
    moments::UniVarMomentsAcc{Tt, Tl, Array}
    nl::Int
    ns::Int

    function SNRMoments{Tt, Tl}(ns::Int, nl::Int) where {Tt<:AbstractFloat, Tl<:Integer}
        moments = UniVarMomentsAcc{Tt, Tl, Array}(2, ns, nl)
        new(moments, nl, ns)
    end
end

mutable struct SNRMomentsChunked{Tt<:AbstractFloat, Tl<:Integer}
    chunksize::Int  # chunks may not be of exactly `chunksize` dim
    chunk_map::Dict{UnitRange, SNRMoments{Tt, Tl}}
    nl::Int
    ns::Int

    function SNRMomentsChunked{Tt, Tl}(ns::Int, nl::Int, chunksize::Int = 16384) where {Tt<:AbstractFloat, Tl<:Integer}
        slices = tiled_view(1:ns, (chunksize, ))
        chunk_map = Dict(slice => SNRMoments{Tt, Tl}(size(slice, 1), nl) for slice in slices)
        new(chunksize, chunk_map, nl, ns)
    end
end

mutable struct SNROrdered{Tt<:AbstractFloat, Tl<:Integer}
    moments::UniVarMomentsAcc{Tt, Tl, Array}
    order::Int

    function SNROrdered{Tt, Tl}(ns::Int, nl::Int, order::Int = 1) where {Tt<:AbstractFloat, Tl<:Integer}
        moments = UniVarMomentsAcc{Tt, Tl, Array}(2*order, ns, nl)
        new(moments, order)
    end
end

function SNR_fit!(snr::SNRBasic{Tt, Tl}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:Real, Tl<:Real}
    samples_per_thread = cld(size(traces, 2), Threads.nthreads())
    trace_tiles = tiled_view(traces, (size(traces, 1), samples_per_thread))
    sum_tiles = tiled_view(snr.sums, (size(traces, 1), samples_per_thread))
    sum_sq_tiles = tiled_view(snr.sums_sq, (size(traces, 1), samples_per_thread))

    if !all(snr.totals == 0)
        throw(ArgumentError("This type of SNR struct may only be fit once, and this instance has already been fit"))
    end
    
    Threads.@threads for (tile_y, (trace_tile, sum_tile, sum_sq_tile)) in collect(enumerate(collect(zip(trace_tiles, sum_tiles, sum_sq_tiles))))
        for trace in axes(trace_tile, 1)
            @inbounds l = Int(labels[trace]) + 1
            if tile_y == 1 
                @inbounds Atomix.@atomic snr.totals[l] += 1 
            end
            @inbounds @views sum_tile[l, :] .+= trace_tile[trace, :]
            @inbounds @views sum_sq_tile[l, :] .+= trace_tile[trace, :].^2
        end
    end
end

function SNR_fit!(snr::Union{SNRMoments{Tt, Tl}, SNROrdered{Tt, Tl}}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:Real, Tl<:Real}
    centered_sum_update!(snr.moments, traces, labels)
end

function SNR_fit!(snr::SNRMomentsChunked{Tt, Tl}, traces::AbstractMatrix{Tt}, labels::AbstractVector{Tl}) where {Tt<:Real, Tl<:Real}
    # TODO: these can be dispatched asynchronously 
    for trace_slice in collect(keys(snr.chunk_map))
        SNR_fit!(snr.chunk_map[trace_slice], traces[:, trace_slice], labels[:])
    end
end

function SNR_finalize(snr::SNRBasic{Tt, Tl})::Vector where {Tt<:Real, Tl<:Real}
    means = snr.sums ./ snr.totals
    signals = var(means, dims=1)

    vars = (snr.sums_sq ./ snr.totals) .- (means.^2)
    noises = mean(vars, dims=1)

    signals ./ noises
end

function SNR_finalize(snr::SNRMoments{Tt, Tl})::Vector where {Tt<:Real, Tl<:Integer}
    means = @views snr.moments.moments[:, 1, :]
    signals = var(means, dims=1)
    
    vars = (snr.moments.moments[:, 2, :] ./ snr.moments.totals)
    noises = mean(vars, dims=1)

    vec(signals ./ noises)
end

function SNR_finalize(snr::SNRMomentsChunked{Tt, Tl})::Vector where {Tt<:Real, Tl<:Integer}
    out = zeros(snr.ns)
    Threads.@threads for slice in collect(keys(snr.chunk_map))
        out[slice] .= SNR_finalize(snr.chunk_map[slice])
    end
    out
end

function SNR_finalize(snr::SNROrdered)::Vector 
    μ, σ2 = get_mean_and_var(snr.moments, snr.order)
    signals = var(μ, dims=1)
    noises = mean(σ2, dims=1)
    signals ./ noises
end


end  # module SNR
# 